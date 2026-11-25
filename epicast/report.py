"""Build the dashboard: one self-contained HTML file with the data embedded.

The dashboard needs no server and no internet connection. Open it by double
click, attach it to an email, or host it on any static site (GitHub Pages, S3).
Re-running the pipeline regenerates it from the latest data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean, config, geo, outlook, registry
from .backtest import Q_COLS
from .models import MODEL_LABELS, ensemble_members
from .models.outbreak import Renewal

REPLAY_Q = ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]  # quantiles kept for backtest replays
# Places outside the province file that the Bundibugyo line list reports: (lat, lon).
BVD_COUNTRY_POINTS = {
    "Uganda": ("UGA", "Kampala", 0.316, 32.582),
    "Kenya": ("KEN", "Nairobi", -1.283, 36.817),
    "Nigeria": ("NGA", "Abuja", 9.077, 7.399),
}
# Registry province names -> ISO 3166-2 codes in the DRC boundary file (used when a row has no coordinates).
REGISTRY_PROVINCE_ISO = {"Bas-Uele": "CD-BU", "Equateur": "CD-EQ", "Kasai": "CD-KS", "Nord-Kivu": "CD-NK",
                         "Ituri": "CD-IT", "Haut-Uele": "CD-HU", "Tshuapa": "CD-TU"}
OUTBREAK_SET_ORDER = ("bvd_weekly_cases", "ebola_weekly_cases")
OUTBREAK_SET_NOTES = {
    "bvd_weekly_cases": ("Bundibugyo virus disease, DR Congo, 2026. This outbreak is still in progress. Weekly "
                         "counts are laboratory-confirmed cases by week of confirmation. The last slider position "
                         "is a real forecast beyond the data. A portfolio exercise, not public-health guidance."),
    "ebola_weekly_cases": ("West Africa, 2014 to 2016. This outbreak is over. It is kept as a test bed: drag the "
                           "slider to any week to see what each model would have forecast with only the data "
                           "available then, next to what happened."),
}


def _sig(x, sig: int = 4):
    """Round to ``sig`` significant figures; NaN and None become None (JSON null)."""
    if x is None or not np.isfinite(x):
        return None
    if x == 0:
        return 0
    mag = int(np.floor(np.log10(abs(x))))
    out = round(float(x), sig - 1 - mag)
    return int(out) if out == int(out) and abs(out) < 1e15 else out


def _matrix(df: pd.DataFrame, cols: list[str]) -> list[list]:
    return [[_sig(v) for v in row] for row in df.sort_values("horizon")[cols].to_numpy(float)]


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: (_sig(v, 4) if isinstance(v, (float, np.floating)) else
                 (int(v) if isinstance(v, (int, np.integer)) else v))
             for k, v in row.items()} for row in df.to_dict("records")]


def _clean_json(obj):
    """Recursively replace NaN with None so the payload is strict JSON."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- parts
def _annual_part(obs, final, back, names) -> dict:
    out = {}
    for key in config.ANNUAL_TARGETS:
        target = config.TARGETS[key]
        series = clean.series_for(obs, target)
        f = final[final["target"] == key]
        b = back[back["target"] == key]
        locs = sorted(f["location"].unique(), key=lambda c: names.get(c, c))
        entry = {
            "label": target.label, "unit": target.unit, "horizons": list(target.horizons),
            "members": list(ensemble_members(target)), "minLevel": target.min_level,
            "lastYear": int(max(s.index[-1] for s in series.values())),
            "locations": [{"id": c, "name": names.get(c, c)} for c in locs],
            "series": {}, "final": {}, "backtest": {},
        }
        for c in locs:
            s = series[c]
            entry["series"][c] = {"x": [int(v) for v in s.index], "y": [_sig(v, 5) for v in s.to_numpy()]}
            entry["final"][c] = {m: _matrix(g, Q_COLS) for m, g in f[f["location"] == c].groupby("model")}
            bc = b[b["location"] == c]
            entry["backtest"][c] = {origin: {m: _matrix(g, REPLAY_Q) for m, g in go.groupby("model")}
                                    for origin, go in bc.groupby("origin")}
        out[key] = entry
    return out


def _outbreak_sets(obs, final, back, names) -> list[dict]:
    renewal = Renewal()
    sets = []
    for key in OUTBREAK_SET_ORDER:
        target = config.TARGETS[key]
        series = clean.series_for(obs, target)
        b = back[back["target"] == key]
        f = final[final["target"] == key]
        entry = {"key": key, "label": target.label, "unit": target.unit, "ongoing": target.ongoing,
                 "horizons": list(target.horizons), "note": OUTBREAK_SET_NOTES[key],
                 "locations": [{"id": c, "name": names.get(c, c)} for c in series],
                 "series": {}, "rt": {}, "backtest": {}, "final": {}}
        for c, s in series.items():
            entry["series"][c] = {"x": [d.strftime("%Y-%m-%d") for d in s.index],
                                  "y": [_sig(v, 5) for v in s.to_numpy()]}
            entry["rt"][c] = [_sig(v, 3) for v in renewal.r_history(s.to_numpy())]
            bc = b[b["location"] == c]
            entry["backtest"][c] = {origin: {m: _matrix(g, REPLAY_Q) for m, g in go.groupby("model")}
                                    for origin, go in bc.groupby("origin")}
            fc = f[f["location"] == c]
            if len(fc):
                entry["final"][c] = {m: _matrix(g, REPLAY_Q) for m, g in fc.groupby("model")}
        sets.append(entry)
    return sets


def _map_part(obs, geo_data, outbreaks, sources, bvd) -> dict:
    """Everything the map needs: outlines, per-country values, outbreak points, proximity."""
    in_map = {c["id"] for c in geo_data["countries"]}
    centroids = geo.province_centroids(geo_data)

    def by_country_year(disease: str) -> dict:
        sub = obs[(obs["disease"] == disease) & obs["location"].isin(in_map)]
        return {loc: {int(p): _sig(v, 4) for p, v in zip(g["period"].astype(int), g["value"])}
                for loc, g in sub.groupby("location")}

    shapes = geo.country_shapes()
    name_of = {c["id"]: c["name"] for c in geo_data["countries"]}
    rows = []
    for _, r in outbreaks.iterrows():
        lat, lon, precision = r["lat"], r["lon"], r["place_precision"]
        if not np.isfinite(lat):  # no town coordinates: fall back to the province centre
            lat, lon = centroids[REGISTRY_PROVINCE_ISO[r["admin1"]]]
        cases, deaths, note = r["cases"], r["deaths"], r["case_note"]
        if r["outbreak_id"] == "cod-2026-ituri":
            cases, deaths = bvd["confirmed"], bvd["deaths_among_confirmed"]
            note = f"Laboratory-confirmed cases in the line list, data to {bvd['as_of']}"
        near = {}
        for iso, shp in shapes.items():
            if iso == r["country"]:
                continue
            km = geo.distance_to_country_km(lat, lon, shp)
            if km <= config.NEARBY_KM:
                near[iso] = int(round(km))
        rows.append({
            "id": r["outbreak_id"], "virus": r["virus"], "virusLabel": registry.VIRUSES[r["virus"]],
            "country": r["country"], "countryName": name_of.get(r["country"], r["country"]),
            "place": r["place"], "precision": precision, "lat": round(float(lat), 3), "lon": round(float(lon), 3),
            "start": r["start"], "end": r["end"], "startYear": int(r["start_year"]),
            "endYear": (int(r["end_year"]) if np.isfinite(r["end_year"]) else None), "status": r["status"],
            "cases": (int(cases) if np.isfinite(cases) else None),
            "deaths": (int(deaths) if np.isfinite(deaths) else None),
            "caseNote": note, "notes": r["notes"], "near": near,
            "sources": registry.source_links(r["source_ids"], sources),
        })

    points = []
    for p in bvd["provinces"]:
        iso2 = config.BVD_PROVINCE_ISO.get(p["name"])
        if iso2 in centroids:
            lat, lon = centroids[iso2]
            points.append({"name": p["name"], "country": "COD", "lat": lat, "lon": lon,
                           "confirmed": p["confirmed"], "deaths": p["deaths"], "precision": "province centre"})
    for c in bvd["countries"]:
        if c["name"] in BVD_COUNTRY_POINTS:
            iso, town, lat, lon = BVD_COUNTRY_POINTS[c["name"]]
            points.append({"name": f"{c['name']} ({town})", "country": iso, "lat": lat, "lon": lon,
                           "confirmed": c["confirmed"], "deaths": c["deaths"], "precision": "capital city"})
    return {
        "countries": geo_data["countries"], "provinces": geo_data["provinces"],
        "malaria": by_country_year("malaria"), "hiv": by_country_year("hiv"),
        "outbreaks": rows, "bvd": {**bvd, "points": points}, "nearbyKm": config.NEARBY_KM,
        "yearMin": 2000, "yearMax": datetime.now(timezone.utc).year,
    }


def _outlook_part(obs, final, overall, outbreaks) -> dict:
    annual = {}
    for key in config.ANNUAL_TARGETS:
        target = config.TARGETS[key]
        series = clean.series_for(obs, target)
        latest = {loc: float(s.iloc[-1]) for loc, s in series.items()}
        model = outlook.best_model(overall, key)
        per = outlook.annual_outlook(final, latest, key, model)
        annual[key] = {"label": target.label, "unit": target.unit, "model": model,
                       "lastYear": int(max(s.index[-1] for s in series.values())),
                       "byCountry": {k: {kk: (_sig(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                     for k, v in per.items()}}
    bvd_target = config.TARGETS["bvd_weekly_cases"]
    bvd_series = clean.series_for(obs, bvd_target)
    f = final[final["target"] == bvd_target.key]
    live = []
    for loc, s in bvd_series.items():
        o = outlook.outbreak_outlook(s, f, loc)
        o["name"] = config.BVD_SERIES[loc][0]
        live.append(o)
    now = pd.Timestamp(datetime.now(timezone.utc).date())
    return {"annual": annual, "recurrence": outlook.recurrence_table(outbreaks, now), "bvd": live,
            "recurrenceSince": config.REGISTRY_START_YEAR}


# ------------------------------------------------------------------------- payload
def build_payload(output_dir: Path | None = None) -> dict:
    out_dir = output_dir or config.OUTPUT_DIR
    obs = clean.load_observations()
    final = pd.read_csv(out_dir / "forecasts.csv", low_memory=False)
    back = pd.read_csv(out_dir / "backtest_forecasts.csv.gz", low_memory=False)
    back["origin"] = back["origin"].astype(str)
    overall = pd.read_csv(out_dir / "metrics_overall.csv")
    by_h = pd.read_csv(out_dir / "metrics_by_horizon.csv")
    sources = registry.load_sources()
    outbreaks = registry.load_outbreaks(sources=sources)
    geo_data = geo.load_geo()
    bvd = json.loads((config.PROCESSED_DIR / "bvd_summary.json").read_text(encoding="utf-8"))

    names = obs.drop_duplicates("location").set_index("location")["location_name"].to_dict()
    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "quantiles": list(config.QUANTILES),
            "replayQuantiles": [float(c[1:]) for c in REPLAY_Q],
            "modelLabels": MODEL_LABELS,
            "sources": sorted(obs["source"].unique().tolist()),
            "sourceVersions": {d: sorted(g["source_version"].astype(str).unique().tolist())
                               for d, g in obs.groupby("disease")},
            "allSources": sources[["title", "publisher", "url"]].to_dict("records"),
        },
        "annual": _annual_part(obs, final, back, names),
        "outbreakSets": _outbreak_sets(obs, final, back, names),
        "map": _map_part(obs, geo_data, outbreaks, sources, bvd),
        "outlook": _outlook_part(obs, final, overall, outbreaks),
        "metrics": {"overall": _records(overall), "byHorizon": _records(by_h)},
        "targetLabels": {k: t.label for k, t in config.TARGETS.items()},
    }
    return _clean_json(payload)


def build_dashboard(output_dir: Path | None = None) -> Path:
    out_dir = output_dir or config.OUTPUT_DIR
    payload = build_payload(out_dir)
    template = (config.TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = template.replace("/*__EPICAST_DATA__*/null", data)
    path = out_dir / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path
