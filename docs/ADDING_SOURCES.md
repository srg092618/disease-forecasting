# Adding sources and further information

Every number on the map that comes from an outbreak report carries a link back to that
report. The links are not written in the dashboard code. They come from two small CSV files
that you edit by hand:

| File | One row is | Used for |
|---|---|---|
| `epicast/reference/sources.csv` | one source: a report, a data set, a paper | the link text and URL |
| `epicast/reference/outbreaks.csv` | one outbreak | the map marker, the outbreak table, the recurrence statistics |

An outbreak points at its sources through the `source_ids` column. That is the whole
mechanism: add a row, re-run one command, and the link appears.

```
outbreaks.csv                      sources.csv
-------------                      -----------
uga-2022-mubende                   who_don433  ->  title, publisher, URL
  source_ids = who_don433;cdc_...  cdc_outbreak_history  ->  title, publisher, URL
```

## 1. Add a source

Open `epicast/reference/sources.csv` and add one line.

| Column | What to put | Example |
|---|---|---|
| `source_id` | a short unique name, lower case, no spaces | `who_don620` |
| `title` | the document's own title | `Disease Outbreak News: Bundibugyo virus disease - Uganda` |
| `publisher` | who published it | `World Health Organization` |
| `url` | the full link, starting with `https://` | `https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON620` |
| `accessed` | the date you read it, `YYYY-MM-DD` | `2026-11-20` |
| `notes` | what you took from it | `Case counts by district as of 18 Nov 2026` |

If a value contains a comma, wrap it in double quotes.

A new source appears straight away in the "All sources" list on the dashboard's
"How to read this" tab, even before any outbreak uses it. Use that for background reading
that is not tied to one outbreak (a review paper, a WHO fact sheet, a methods note).

## 2. Attach a source to an outbreak

Open `epicast/reference/outbreaks.csv`, find the outbreak's row, and add your `source_id` to
its `source_ids` column. Separate several ids with semicolons, most important first:

```
who_don433;cdc_outbreak_history;who_don620
```

The first source is the one a reader sees first in the outbreak table.

## 3. Add a new outbreak

Add one line to `outbreaks.csv`.

| Column | Rule |
|---|---|
| `outbreak_id` | unique; the pattern is `country-year-place`, for example `uga-2027-gulu` |
| `virus` | `EBOV` (Ebola virus), `SUDV` (Sudan virus) or `BDBV` (Bundibugyo virus) |
| `country` | ISO3 code of the country where it was declared: `COD`, `UGA` |
| `admin1` | province, if known |
| `place` | the place name as the source gives it |
| `place_precision` | what the coordinates stand for: `town (Gulu)`, `province centroid`, `epicentre town (Beni)` |
| `lat`, `lon` | decimal degrees. Leave both empty for a DRC outbreak to use the centre of the province in `admin1` |
| `start`, `end` | `YYYY`, `YYYY-MM` or `YYYY-MM-DD`, as precise as the source is. Leave `end` empty while it is ongoing |
| `status` | `ended` or `ongoing` |
| `cases`, `deaths` | whole numbers; leave empty if not reported |
| `case_note` | how cases are defined, for example `12 confirmed and 2 probable` |
| `source_ids` | at least one id from `sources.csv` |
| `notes` | anything a reader should know |

Write down in `place_precision` how you got the coordinates. A marker that looks exact but
is really a province centre misleads people.

The recurrence figures on the outlook cards ("46% chance within 12 months") are computed
from this file, so a new row changes them on the next run.

## 4. Check and rebuild

```bash
pytest tests/test_registry_geo_outlook.py     # the registry rules
python -m epicast report                      # rebuild outputs/dashboard.html
```

The registry is validated every time it is loaded. These mistakes stop the run with a
message that names the row:

* a `source_id` used by an outbreak but missing from `sources.csv`
* a source without an `http(s)` link or without a title
* an outbreak with no source, no start, or an end before its start
* more deaths than cases
* a duplicate id

Then open the dashboard and check three places: the marker's detail panel (click the
marker), the outbreak table under the map, and the "All sources" list.

## 5. Keeping the live outbreak current

The Bundibugyo counts come from the Global.health line list, pinned to one commit so that
a run can be reproduced.

* One-off refresh with the newest data: `python -m epicast run --live-outbreak --jobs 2`
* To move the pin: set `BVD_LINELIST_COMMIT` in `epicast/config.py` to a newer commit hash
  from <https://github.com/globaldothealth/outbreak-data/commits/main>, then run the pipeline.

When the outbreak is declared over, set its `end` and `status` in `outbreaks.csv` and set
`ongoing=False` on the `bvd_weekly_cases` target in `config.py`.

## 6. Ideas for going further

These are not built yet. Each is a contained piece of work.

**Links per country or per disease.** Add a file `epicast/reference/further_reading.csv`
with columns `scope` (an ISO3 code or a disease name), `title`, `publisher`, `url`.
Load and validate it in `registry.py` the same way as `sources.csv`. In `report.py`, add it
to the payload inside `_map_part`. In `templates/dashboard.html`, show the matching rows in
`renderOutlook()` under the outlook cards for the focused country. The helper
`sourceLinks(host, sources)` already turns a list of `{title, publisher, url}` into links.

**Health-zone positions for Bundibugyo.** The map places 2026 cases by province because the
project has no coordinates for health zones. Add `epicast/reference/health_zones.csv`
(`name`, `province`, `lat`, `lon`, `source_id`), join it to `bvd_summary()["health_zones"]`
in `report.py`, and add the points to `bvd["points"]`. The map draws whatever is in that list.

**Link each marker to a specific situation report.** The line list has a `Source_I` column
with the report each case came from. `clean.bvd_summary` could return the most recent
report URL per province and show it in the marker's detail panel.

**Cite a number, not just an outbreak.** Add a `source_id` column next to any figure you
quote in a README or write-up, and keep the `accessed` date. Numbers for an ongoing
outbreak change weekly; the date is what makes a figure checkable later.
