# disease-forecasting (`epicast`)

A tested forecasting pipeline and interactive map for malaria, HIV and three Ebola-disease
viruses in Africa. It downloads public data, cleans it into one tidy table, forecasts with
eight model families, scores every model against history it had not seen, recalibrates the
uncertainty intervals, and writes a single-file dashboard with a zoomable map.

This is the **reference implementation**. A separate repository,
[`disease-forecasting-learn`](https://github.com/srg092618/disease-forecasting-learn), is a
staged course for building your own version from scratch. The two repos share no code.

> **Not for operational use.** This is a portfolio and learning project. Bundibugyo virus
> disease in DR Congo is a live public-health emergency; for decisions, use WHO, Africa CDC
> and the DRC Ministry of Health.

## Quick start

```bash
git clone https://github.com/srg092618/disease-forecasting.git
cd disease-forecasting
python -m venv .venv
source .venv/bin/activate            # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt

pytest                               # 83 tests, about 20 seconds, no network needed
python -m epicast run --jobs 2       # ingest -> clean -> backtest -> forecast -> report (about 3 minutes)
```

Open `outputs/dashboard.html` in a browser. It needs no server and no internet.

| Command | What it does |
|---|---|
| `python -m epicast run` | All five steps from the pinned, reproducible data |
| `python -m epicast run --live-outbreak` | Same, but reads the newest Bundibugyo line list instead of the pinned commit |
| `python -m epicast run --source who` | Malaria and HIV from the live WHO API instead of the pinned mirror |
| `python -m epicast run --skip-ingest` | Reuse raw files already on disk |
| `python -m epicast ingest \| clean \| backtest \| forecast \| report` | One step at a time |

## What it covers

| Section | Data | Frequency | Forecast horizon |
|---|---|---|---|
| Malaria incidence per 1,000 at risk | WHO estimates, 107 countries, 2000 to 2024 | annual | 1 to 5 years |
| New HIV infections | UNAIDS estimates, 147 countries, 1990 to 2024 | annual | 1 to 5 years |
| **Ebola virus (Zaire), DR Congo** | Outbreak registry: 12 DRC outbreaks, 2007 to 2025 | per outbreak | 12-month recurrence chance |
| **Sudan virus, Uganda** | Registry: 5 outbreaks, incl. Mubende 2022–23 and Kampala 2025 | per outbreak | 12-month recurrence chance |
| **Bundibugyo virus, all regions** | Registry (Uganda 2007, DRC 2012, DRC 2026) plus the 2026 line list | weekly for 2026 | 1 to 4 weeks |
| Ebola, West Africa 2014–16 | Situation-report transcriptions, 3 countries | weekly | 1 to 4 weeks (kept as the long benchmark) |

## The dashboard

* **Map and outlook** (default tab). Africa map with zoom (buttons, wheel, drag) and hover.
  Pick a focus country to frame it and its neighbours. From/To sliders and presets choose
  the years. Layers: malaria as country shading, HIV as circles, and Ebola-disease
  outbreaks as circle (Ebola virus), square (Sudan virus) and diamond (Bundibugyo virus),
  sized by cases. "Hotspots" fades low values so the high ones stand out. Outbreaks near
  a border show their distance to the neighbouring country. Click a marker for its sources.
  Beside the map: outlook cards ("likely to fall / rise / no clear direction", outbreak
  growth with R, and 12-month recurrence chances), a metrics table for the focus area, an
  outbreak table with sources, and a Bundibugyo health-zone table.
* **Malaria and HIV**: per-country history and forecast fans.
* **Outbreak forecasts**: Bundibugyo 2026 (DRC and Ituri, Nord-Kivu, Haut-Uele) and West
  Africa 2014–16, with a slider that replays what each model would have said each week.
  The last slider position is the live forecast.
* **Accuracy**: backtest scores. **How to read this**: method notes and every source.

## Data sources

All downloads are pinned to fixed commits, stored untouched with a UTC timestamp, and
logged with their SHA-256 hash in `data/raw/manifest.jsonl`.

| Source | Used for | Pin |
|---|---|---|
| open-numbers mirror of World Bank WDI (republishes WHO malaria and UNAIDS HIV estimates) | Malaria, HIV | `1bd7fac42b` |
| WHO Global Health Observatory API (`--source who`) | Malaria, HIV (live) | not pinned |
| cmrivers/ebola and imdevskp transcriptions of WHO situation reports | West Africa 2014–16 | `ddcbf44`, `c23b14a` |
| Global.health Bundibugyo line list | BVD 2026 weekly confirmed cases | `4bc4b57` |
| CDC outbreak history, WHO Disease Outbreak News, WHO AFRO, ECDC | Outbreak registry | per row in `epicast/reference/sources.csv` |
| Natural Earth admin-0 and admin-1 | Country and Uganda district shapes | `ca96624a` |
| geoBoundaries COD ADM1 | DRC province shapes | `8ce27006` |

Every outbreak row in `epicast/reference/outbreaks.csv` lists its source ids; the pipeline
refuses to run if a source id is unknown, a URL is missing, deaths exceed cases, or a
coordinate is off the map. To add sources or outbreaks, see
[`docs/ADDING_SOURCES.md`](docs/ADDING_SOURCES.md).

## Models

| Name | Family | Used for |
|---|---|---|
| `naive` | Random walk: next value equals the last. The benchmark. | all |
| `drift` | Random walk with drift | annual |
| `ets` | Exponential smoothing, trend chosen by AICc | all |
| `arima` | ARIMA(p,1,q) with drift, chosen by AICc | annual |
| `gbm` | One gradient-boosting model across all countries, quantile loss | annual |
| `renewal` | Renewal equation with time-varying R (Cori method), negative-binomial noise | outbreaks |
| `richards` | Richards growth curve with parametric bootstrap | outbreaks |
| `ensemble` | Median of members at every quantile; members fixed in advance | all |

Every model returns nine quantiles (2.5% to 97.5%), on z = log(1 + y).

**Recurrence chance.** New outbreaks are treated as a Poisson process. With N outbreaks in T
years and a Jeffreys prior, the chance of at least one in the next 12 months is
1 − (T / (T + 1))^(N + ½). This is a historical base rate, not a prediction of where or when.

**Outlook direction.** "Likely to fall" means the whole 80% forecast interval is below the
latest value; "likely to rise" means it is entirely above; otherwise "no clear direction".

## Evaluation

* **Rolling origin.** For each past year or week: hide the future, fit, forecast, compare.
  A test scrambles the future and checks the forecasts do not change.
* **Weighted interval score (WIS)** on the log scale, reported relative to naive (below 1.00 is better).
* **Coverage**: how often each interval contained the outcome.
* **Recalibration**: conformalized quantile regression with a volatility-scaled margin,
  using only outcomes known at the forecast origin. Margins never shrink with horizon.

## Results (calibrated intervals)

| Section | Best model | Relative error vs naive | 95% coverage |
|---|---|---|---|
| New HIV infections (6,499 forecasts per model) | Global gradient boosting | 0.58 (95% CI 0.54 to 0.63) | 94% |
| Malaria incidence (3,586) | Naive; nothing beats it | GBM 1.01 (0.94 to 1.07) | 94% |
| Ebola, West Africa (776) | Naive; ensemble next | ensemble 1.08 | 94% |
| Bundibugyo 2026 (256) | Renewal equation | renewal 0.94, ensemble 0.97 | 98%, 99% |

Example forecast: DR Congo malaria incidence in 2029, gradient boosting, median 299 per
1,000 at risk, 95% interval 190 to 389 (in `outputs/forecasts.csv`).

What this says:

1. **HIV**: pooled machine learning cuts error by about 40%; UNAIDS estimates are smooth.
2. **Malaria**: nothing beats "next year equals this year"; incidence has stalled since 2015.
3. **Outbreaks**: at 1 to 4 weeks, beating persistence is hard. On Bundibugyo the renewal
   model edges ahead, but with only 256 forecasts the margin is not conclusive.
4. **Calibration works**: after recalibration, 95% intervals cover about 93% to 99%. The
   Richards curve stays under-covered on Bundibugyo (82%).

Full tables: `outputs/metrics_overall.csv` and `outputs/metrics_by_horizon.csv`.

## Limitations

* **Bundibugyo is ongoing.** Counts are confirmed cases by confirmation date; the last,
  partial week is dropped. Late reports will revise recent weeks.
* **Borrowed serial interval.** R uses the West Africa Ebola serial interval (gamma, mean
  15.3, SD 9.3 days). Bundibugyo and Sudan virus may differ.
* **Positions are approximate.** Outbreaks sit at the town, district or province centre;
  Bundibugyo cases are placed by province because health-zone coordinates are not in the
  pinned data. Health zones are listed in a table instead.
* **Estimates get revised.** WHO and UNAIDS re-estimate history every year; the backtest
  uses one vintage, which flatters it.
* **The backtest guided development**, so scores are somewhat optimistic.
* **No covariates** (funding, bed nets, rainfall, conflict, vaccination).
* **Recurrence assumes a constant rate**, ignoring surveillance improvements and vaccination.

## Layout

```
epicast/
  config.py        paths, pinned sources, targets, quantile grid
  ingest.py        downloads (timestamped and hashed), incl. boundaries and BVD line list
  schema.py        the tidy table and its validation
  clean.py         raw -> tidy; BVD weekly series and summary
  registry.py      outbreak and source registry, validated
  geo.py           shapes, neighbours, distances (shapely, haversine)
  outlook.py       recurrence chances and at-a-glance outlook
  models/          statistical, pooled, outbreak, ensemble
  metrics.py       pinball loss, WIS, coverage, MASE
  backtest.py      rolling-origin evaluation
  calibrate.py     conformal recalibration
  pipeline.py      the five steps
  report.py        builds the dashboard
  reference/       outbreaks.csv, sources.csv, countries.csv
  templates/dashboard.html
tests/             83 tests
docs/              ADDING_SOURCES.md
outputs/           metrics, forecasts, dashboard (committed so results are inspectable)
legacy/            the original ILINet scaffold, kept for history
data/              raw and processed data (git-ignored, re-created by the pipeline)
```

## Automation

`.github/workflows/ci.yml` runs the tests on every push and pull request, and every Monday
(or on demand) rebuilds the dashboard with `--live-outbreak` and uploads it as a build
artifact. It has not yet run on GitHub; check the first run.
