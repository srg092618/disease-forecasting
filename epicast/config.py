"""Central configuration: paths, data sources, forecast targets and the quantile grid.

Everything that a reader might want to change lives here, so the rest of the
package contains logic only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- paths
STACK_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = STACK_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # untouched downloads, timestamped, never edited
PROCESSED_DIR = DATA_DIR / "processed"  # tidy tables produced by clean.py
OUTPUT_DIR = STACK_ROOT / "outputs"   # metrics, forecasts and the dashboard
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

REQUEST_TIMEOUT_SECONDS = 60

# ------------------------------------------------------------------ quantile grid
# Every forecast is a set of quantiles, not a single number. This grid gives a
# median plus four central prediction intervals (50%, 80%, 90%, 95%).
QUANTILES: tuple[float, ...] = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
INTERVALS: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)  # nominal central coverage
MEDIAN_INDEX = QUANTILES.index(0.50)

# ------------------------------------------------------------------- data sources
# 1) Default, reproducible source for malaria and HIV: a pinned commit of the
#    Gapminder "open-numbers" mirror of the World Bank World Development
#    Indicators (WDI). WDI republishes WHO Global Health Observatory malaria
#    estimates and UNAIDS HIV estimates. Pinning the commit means the pipeline
#    returns byte-identical data every time, on any machine and in CI.
WDI_MIRROR_COMMIT = "1bd7fac42b3d5012353607bf2464a485557fd049"  # auto-generated 2026-10-18
WDI_MIRROR_BASE = (
    "https://raw.githubusercontent.com/open-numbers/"
    f"ddf--open_numbers--world_development_indicators/{WDI_MIRROR_COMMIT}"
)

# 2) Live source: the WHO Global Health Observatory OData API (no key needed).
#    Newer than the mirror (it carries the latest estimation round) but it
#    changes whenever WHO/UNAIDS revise their estimates.
WHO_GHO_BASE = "https://ghoapi.azureedge.net/api"

# 3) Ebola, West Africa 2014-2016. Two public transcriptions of WHO / Ministry of
#    Health situation reports, pinned to fixed commits:
#    - C. Rivers et al.: cumulative cases from the first reports (Mar 2014) on.
#    - WHO situation-report series as published on the Humanitarian Data
#      Exchange (HDX), mirrored on GitHub: 29 Aug 2014 to 23 Mar 2016.
EBOLA_EARLY_URL = (
    "https://raw.githubusercontent.com/cmrivers/ebola/"
    "ddcbf44475a84b324ad8ca0c075784d070e70d37/country_timeseries.csv"
)
EBOLA_WHO_URL = (
    "https://raw.githubusercontent.com/imdevskp/ebola_outbreak_dataset/"
    "c23b14a46393e512d82eb1284f7f6949ed2efcf3/ebola_2014_2016_raw.csv"
)
EBOLA_COUNTRIES = {"GIN": "Guinea", "LBR": "Liberia", "SLE": "Sierra Leone"}

# 4) Bundibugyo virus disease, DRC and neighbours, 2026 (ongoing). Case-level line
#    list curated by Global.health from official situation reports. The default
#    is a pinned commit so that a run is reproducible; `--live-outbreak` reads the
#    newest version instead.
BVD_LINELIST_COMMIT = "4bc4b57"  # 2026-11-16
BVD_LINELIST_PATH = "Ebola%20BVD/Data/Ebola%20BVD%202026%20linelist%20-%20PUBLIC%20VIEW.csv"
BVD_LINELIST_URL = ("https://raw.githubusercontent.com/globaldothealth/outbreak-data/"
                    "{ref}/" + BVD_LINELIST_PATH)
# Series forecast for this outbreak: the national total and the three provinces
# with sustained transmission. Keys are location codes, values are (label, how
# the line list names the place).
BVD_SERIES = {
    "COD": ("DR Congo (all provinces)", None),
    "COD-IT": ("Ituri", "Ituri"),
    "COD-NK": ("Nord-Kivu", "Nord-Kivu"),
    "COD-HU": ("Haut-Uele", "Haut-Uele"),
}
# Line-list province names -> ISO 3166-2 codes used by the boundary file.
BVD_PROVINCE_ISO = {
    "Ituri": "CD-IT", "Nord-Kivu": "CD-NK", "Haut-Uele": "CD-HU", "Tshopo": "CD-TO",
    "Bas Uele": "CD-BU", "Nord Ubangi": "CD-NU", "Sud-Kivu": "CD-SK", "Sud Ubangi": "CD-SU",
}

# 5) Map boundaries, pinned. Natural Earth is public domain; geoBoundaries is CC BY.
NATURAL_EARTH_COMMIT = "ca96624a56bd078437bca8184e78163e5039ad19"
NATURAL_EARTH_BASE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                      f"{NATURAL_EARTH_COMMIT}/geojson")
GEOBOUNDARIES_COD_ADM1_URL = ("https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/"
                              "8ce2700661a28ee944350e2b329bbabb58d2ff83/releaseData/gbOpen/COD/ADM1/"
                              "geoBoundaries-COD-ADM1_simplified.geojson")
MAP_CONTINENT = "Africa"
NEARBY_KM = 300.0  # an outbreak this close to a country's border counts as "nearby"
EBOLA_SPLICE_DATE = "2014-08-29"  # first date of the WHO/HDX series


# ------------------------------------------------------------------------ targets
@dataclass(frozen=True)
class Target:
    """One thing we forecast.

    Attributes
    ----------
    key : short identifier used in file names and the CLI.
    disease, metric : what is being measured.
    freq : "A" for annual country estimates, "W" for weekly outbreak counts.
    unit : human-readable unit for charts.
    horizons : how many steps ahead we forecast (years or weeks).
    min_level : a series is "active" at a forecast origin only if its latest
        value is at least this large. Forecasting a country that has eliminated
        malaria (a row of zeros) is trivial and would flatter every model.
    min_train : minimum number of observations before the first forecast origin.
    first_origin : earliest forecast origin used in the backtest (year), or None.
    wdi_code, gho_code : indicator codes in the two annual data sources.
    """

    key: str
    disease: str
    metric: str
    freq: str
    unit: str
    label: str
    horizons: tuple[int, ...]
    min_level: float
    min_train: int
    first_origin: int | None = None
    wdi_code: str | None = None
    gho_code: str | None = None
    ongoing: bool = False  # True for an outbreak still in progress: forecast beyond the data


TARGETS: dict[str, Target] = {
    "malaria_incidence": Target(
        key="malaria_incidence",
        disease="malaria",
        metric="incidence_per_1000_at_risk",
        freq="A",
        unit="cases per 1,000 population at risk",
        label="Malaria incidence",
        horizons=(1, 2, 3, 4, 5),
        min_level=1.0,
        min_train=12,
        first_origin=2011,
        wdi_code="SH.MLR.INCD.P3",
        gho_code="MALARIA_EST_INCIDENCE",
    ),
    "hiv_new_infections": Target(
        key="hiv_new_infections",
        disease="hiv",
        metric="new_infections",
        freq="A",
        unit="people newly infected per year",
        label="New HIV infections",
        horizons=(1, 2, 3, 4, 5),
        min_level=1000.0,
        min_train=15,
        first_origin=2005,
        wdi_code="SH.HIV.INCD.TL",
        gho_code="HIV_0000000026",
    ),
    "ebola_weekly_cases": Target(
        key="ebola_weekly_cases",
        disease="ebola",
        metric="weekly_cases",
        freq="W",
        unit="new cases per week",
        label="Ebola weekly cases (West Africa 2014-16)",
        horizons=(1, 2, 3, 4),
        min_level=10.0,   # at least 10 cases over the last four weeks at the origin
        min_train=10,
    ),
    "bvd_weekly_cases": Target(
        key="bvd_weekly_cases",
        disease="bundibugyo",
        metric="weekly_confirmed_cases",
        freq="W",
        unit="newly confirmed cases per week",
        label="Bundibugyo weekly confirmed cases (DRC 2026)",
        horizons=(1, 2, 3, 4),
        min_level=10.0,
        min_train=8,
        ongoing=True,
    ),
}

ANNUAL_TARGETS = tuple(k for k, t in TARGETS.items() if t.freq == "A")
OUTBREAK_TARGETS = tuple(k for k, t in TARGETS.items() if t.freq == "W")

# Outbreak registry window used for recurrence statistics.
REGISTRY_START_YEAR = 2007

# Ebola serial interval (time between symptom onset in a case and in the person
# they infect): gamma distribution, mean 15.3 days, SD 9.3 days.
# Source: WHO Ebola Response Team, N Engl J Med 2014;371:1481-95.
EBOLA_SERIAL_INTERVAL_MEAN_DAYS = 15.3
EBOLA_SERIAL_INTERVAL_SD_DAYS = 9.3
# No serial-interval estimate specific to Bundibugyo virus is built in; the
# renewal model reuses the West Africa value above. Treat R for Bundibugyo as
# conditional on that assumption.

RANDOM_SEED = 20261116
