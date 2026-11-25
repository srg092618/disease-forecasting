"""Map geometry: country and province outlines, neighbours, distances.

Everything the map needs is prepared here once, in Python, so the dashboard
itself stays simple: it receives ready-made outlines and neighbour lists.

Coordinates are plain longitude and latitude. Africa straddles the equator,
where one degree of longitude and one degree of latitude are almost the same
length, so the map draws them as x and y without a projection library.

Distances
---------
``haversine_km`` is the great-circle distance between two points. Distance
from a point to a country is measured to the nearest point of its outline.
Near the equator one degree is about 111 km in every direction, so the search
for that nearest point is done in degrees and the result converted with the
haversine formula.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from shapely.geometry import Point, mapping, shape
from shapely.ops import nearest_points

from . import config, ingest

EARTH_RADIUS_KM = 6371.0
COUNTRY_TOLERANCE_DEG = 0.03   # about 3 km: invisible at map scale, 10x smaller file
PROVINCE_TOLERANCE_DEG = 0.02
BORDER_BUFFER_DEG = 0.05       # outlines from different sources rarely touch exactly


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _rings(geom) -> list[list[list[float]]]:
    """Outer rings of a (multi)polygon as lists of [lon, lat], rounded to about 1 km."""
    polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    out = []
    for poly in polys:
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        out.append([[round(x, 2), round(y, 2)] for x, y in poly.exterior.coords])
    return out


def _feature(geom, tolerance: float) -> dict:
    simple = geom.simplify(tolerance, preserve_topology=True)
    point = geom.representative_point()  # always inside the shape, unlike a centroid
    minx, miny, maxx, maxy = geom.bounds
    return {"rings": _rings(simple), "centroid": [round(point.x, 3), round(point.y, 3)],
            "bbox": [round(minx, 2), round(miny, 2), round(maxx, 2), round(maxy, 2)]}


def build_geo(raw_dir: Path | None = None) -> dict:
    """Read the raw boundary files and return the compact structure the map uses."""
    countries_raw = json.loads(ingest.latest_raw("ne_countries", raw_dir).read_text(encoding="utf-8"))
    admin1_raw = json.loads(ingest.latest_raw("ne_admin1", raw_dir).read_text(encoding="utf-8"))
    cod_raw = json.loads(ingest.latest_raw("cod_provinces", raw_dir).read_text(encoding="utf-8"))

    shapes: dict[str, object] = {}
    countries = []
    for f in countries_raw["features"]:
        p = f["properties"]
        if p.get("CONTINENT") != config.MAP_CONTINENT:
            continue
        iso = p.get("ISO_A3_EH")
        iso = iso if iso and iso != "-99" else p["ADM0_A3"]
        geom = shape(f["geometry"]).buffer(0)
        shapes[iso] = geom
        countries.append({"id": iso, "name": p["NAME_LONG"], **_feature(geom, COUNTRY_TOLERANCE_DEG)})

    buffered = {iso: g.buffer(BORDER_BUFFER_DEG) for iso, g in shapes.items()}
    for c in countries:
        c["neighbors"] = sorted(o for o, g in shapes.items() if o != c["id"] and buffered[c["id"]].intersects(g))

    provinces = {"COD": [], "UGA": []}
    for f in cod_raw["features"]:
        p = f["properties"]
        provinces["COD"].append({"id": p["shapeISO"], "name": p["shapeName"],
                                 **_feature(shape(f["geometry"]).buffer(0), PROVINCE_TOLERANCE_DEG)})
    for f in admin1_raw["features"]:
        p = f["properties"]
        if p.get("adm0_a3") == "UGA":
            provinces["UGA"].append({"id": p.get("iso_3166_2") or p["name"], "name": p["name"],
                                     **_feature(shape(f["geometry"]).buffer(0), PROVINCE_TOLERANCE_DEG)})
    return {"countries": sorted(countries, key=lambda c: c["name"]), "provinces": provinces}


def country_shapes(raw_dir: Path | None = None) -> dict[str, object]:
    """Full-resolution country shapes keyed by ISO3, for distance calculations."""
    raw = json.loads(ingest.latest_raw("ne_countries", raw_dir).read_text(encoding="utf-8"))
    out = {}
    for f in raw["features"]:
        p = f["properties"]
        if p.get("CONTINENT") == config.MAP_CONTINENT:
            iso = p.get("ISO_A3_EH")
            out[iso if iso and iso != "-99" else p["ADM0_A3"]] = shape(f["geometry"]).buffer(0)
    return out


def distance_to_country_km(lat: float, lon: float, country_shape) -> float:
    """Distance from a point to a country: 0 inside it, else to the nearest border point."""
    pt = Point(lon, lat)
    if country_shape.contains(pt):
        return 0.0
    nearest = nearest_points(country_shape, pt)[0]
    return haversine_km(lat, lon, nearest.y, nearest.x)


def province_centroids(geo: dict) -> dict[str, tuple[float, float]]:
    """``{province id: (lat, lon)}`` for the provinces in the geo structure."""
    return {p["id"]: (p["centroid"][1], p["centroid"][0]) for plist in geo["provinces"].values() for p in plist}


def save_geo(geo: dict, processed_dir: Path | None = None) -> Path:
    processed_dir = processed_dir or config.PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / "geo.json"
    path.write_text(json.dumps(geo, separators=(",", ":")), encoding="utf-8")
    return path


def load_geo(processed_dir: Path | None = None) -> dict:
    path = (processed_dir or config.PROCESSED_DIR) / "geo.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m epicast clean` first.")
    return json.loads(path.read_text(encoding="utf-8"))
