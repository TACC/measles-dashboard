# Project: Update state MMR vaccination-rate data for epiENGAGE Measles Dashboard

## Background

The dashboard at https://epiengage-measles.tacc.utexas.edu/ has a "VACCINE RATES"
section that shows school-level MMR vaccination rate data for a number of US states.
The underlying data lives in the GitHub repo:

- Repo: https://github.com/TACC/measles-dashboard (branch: `development`, also on `main`)
- Data folder: `state_data/` — one CSV per state, named `{STATE_ABBR}_MMR_vax_rate.csv`
- Validator script: `vax_rate_csv_checker.py` (repo root) — **run this against every
  output file before considering it done.**

The existing data in that folder is 1-2 years old. The goal is to re-download the
most current data from each state's public health department and reformat it to
match the existing schema, so the dashboard can be refreshed.

## Target schema

Every state CSV must have exactly these columns (see `vax_rate_csv_checker.py` for
the authoritative rules):

| Column | Notes |
|---|---|
| `Facility Number` | Often blank/NaN — states rarely provide this |
| `School Type` | e.g. `Public`, `Non-public` |
| `School District or Name` | School (or district) name |
| `Facility Address` | Street address |
| `County` | County name (some states report a different geography — see per-state notes) |
| `MMR Vaccination Rate` | Float, 0–100 (not a decimal fraction). No zero values. |
| `Age Group` | e.g. `Kindergarten`, `7th Grade` — depends what the state publishes |

Validator also checks: no duplicate School + Age Group + County combos, and rejects
non-numeric rate values. Suppressed/non-reporting rows (states mark these `DS`,
`DNC`, etc.) should generally be **dropped**, not coerced to 0.

## States in scope

Only states with a **public data source link** on the dashboard are being refreshed
(19 of 24). Five states — **Alabama, Idaho, Iowa, Michigan, New Mexico** — were
originally provided directly via email/data request, not a public page, so they are
out of scope for this refresh (no update available without re-contacting the state).

## Per-state source links and status (as of Aug 2026 research)

| State | Link | Status / format notes |
|---|---|---|
| AZ | https://www.azdhs.gov/preparedness/epidemiology-disease-control/immunization/index.php#reports-immunization-coverage | ⚠️ No bulk school-level download found — public page is an interactive query app (https://app.azdhs.gov/IDRReportStats), county-level PDF summaries only. May need a records request like the original data. |
| CA | https://www.cdph.ca.gov/Programs/CID/DCDC/CDPH%20Document%20Library/Immunization/2023_24CAKindergartenGradeData_Letter.xlsx | Search CDPH site for current-year "Kindergarten Immunization Data" letter/xlsx if this exact link is stale |
| CO | https://data-cdphe.opendata.arcgis.com/datasets/cdphe-colorado-school-and-child-care-immunization-facility-data | ArcGIS Hub — 2025-2026 data published; use Download → CSV on the page |
| CT | Kindergarten: `https://data.ct.gov/api/views/iux5-vrzq/rows.csv?accessType=DOWNLOAD`  7th grade: `https://data.ct.gov/api/views/rz57-x4bb/rows.csv?accessType=DOWNLOAD` | ✅ Direct CSV, stable Socrata URLs (no need to edit URL for new year). Raw columns: `School Name, School Type, Address, City, Zipcode, Planning Region, Polio, DTaP, MMR, HepB, Varicella, HepA, All, Ex_Rel, Ex_Med, Ex_Tot`. Note: CT switched from 8 counties to 9 "planning regions" in 2022 — existing repo file uses old county names, so `Planning Region` needs mapping to `County` or accepted as-is (confirm with user). Suppressed values marked `DS` (data suppressed, <30 students) or `DNC` (did not comply) — drop these rows. |
| IN | https://hub.mph.in.gov/dataset/immunization-division-s-school-supplemental-dashboard | CKAN portal, one-click resource download. 2024-2025 covers K, 1st, 6th, 7th, 12th grade — filter to the age groups the target schema needs |
| KS | https://www.kdhe.ks.gov/2016/Kindergarten-Immunization-Data | Not yet verified this session — check for direct file link on page |
| KY | https://healthtracking.ky.gov/Topics/immunizations/Pages/childhood-immunizations.aspx | Not yet verified this session |
| LA | https://analytics.la.gov/t/LDH/views/SchoolImmunizationDashboard/Kindergarten | Tableau dashboard — use dashboard's own crosstab/download button, not a direct URL |
| ME | https://www.maine.gov/dhhs/mecdc/infectious-disease/immunization/publications/index.shtml | Not yet verified this session |
| MD | https://health.maryland.gov/phpa/OIDEOR/IMMUN/Pages/Kindergarten_Immunization_Rates_by_School.aspx | Not yet verified this session |
| MA | https://www.mass.gov/info-details/school-immunizations | Not yet verified this session — mass.gov typically publishes dated Excel files per grade |
| MN | https://www.health.state.mn.us/people/immunize/stats/school/index.html | Confirmed page structure: separate downloadable CSVs for Kindergarten and 7th grade, county/district/school level, plus an archive of prior years |
| NY | https://health.data.ny.gov/Health/School-Immunization-Survey-Beginning-2019-20-Schoo/btkd-y8bp/about_data | Socrata portal like CT — direct CSV likely available at `https://health.data.ny.gov/resource/btkd-y8bp.csv` or the `/api/views/btkd-y8bp/rows.csv?accessType=DOWNLOAD` pattern |
| NC | https://www.dph.ncdhhs.gov/programs/epidemiology/immunization/data/kindergarten-dashboard | Not yet verified this session |
| OR | https://www.oregon.gov/oha/PH/PREVENTIONWELLNESS/VACCINESIMMUNIZATION/GETTINGIMMUNIZED/Pages/SchRateMap.aspx | Interactive map/dashboard — check for underlying data export |
| PA | https://www.pa.gov/agencies/health/programs/immunizations/rates.html | Not yet verified this session |
| TX | https://www.dshs.texas.gov/immunizations/data/school/coverage | ⚠️ Public page only has statewide/county-level PDF summary reports (Annual Report of Immunization Status). The school-level file in the existing repo almost certainly came from a Texas Public Information Act request, not this page. Likely needs a new records request to refresh. |
| WA | https://doh.wa.gov/data-and-statistical-reports/washington-tracking-network-wtn/school-immunization/dashboard | Not yet verified this session |
| WI | https://www.dhs.wisconsin.gov/library/collection/p-01892 | Not yet verified this session |

## Workflow

1. For each state, download the current raw file (CSV or XLSX — no PDFs needed,
   confirmed all sources are tabular).
2. Write/reuse a transform script per state (or a shared script with per-state
   column-mapping configs) to reshape into the target schema above.
3. Drop suppressed/non-reporting rows rather than coercing to 0.
4. Run `vax_rate_csv_checker.py` against each output file and fix any validation
   errors.
5. Save finished file as `state_data/{STATE_ABBR}_MMR_vax_rate.csv`, matching
   existing naming convention.
6. Flag any state where the public source no longer provides school-level data
   (currently: AZ, TX) rather than silently downgrading granularity.

## Open questions to resolve with the user

- CT: how to handle the county → planning region change (keep `Planning Region`
  as-is in the `County` column, or map back to legacy counties)?
- AZ / TX: whether to pursue a new data request, use county/state-level summary
  data instead, or leave these two states un-updated this cycle.
- Whether to keep 7th grade data where available (some existing state files include
  it, e.g. CT) or standardize to Kindergarten-only.
