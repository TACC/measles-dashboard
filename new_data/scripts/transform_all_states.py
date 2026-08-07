"""Transform newly-downloaded raw state files (new_data/state_data/) into the target
MMR vaccination-rate schema and write them to state_data/.

See new_data/measles_dashboard_vax_data_update.md for background.

Usage:
    python3 new_data/scripts/transform_all_states.py            # all states
    python3 new_data/scripts/transform_all_states.py AZ CA CO    # just these states

Run from anywhere in the repo (paths are resolved relative to this file).
Each named state must have its raw file(s) already present in new_data/state_data/,
matching the filenames hardcoded in its transform_xx() function below.
"""
import os
import re
import subprocess
import sys
from io import StringIO

import numpy as np
import pandas as pd

NEW_DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(NEW_DATA_ROOT)
# Raw downloads from each state, plus the KS district->county lookup.
NEW_DATA = os.path.join(NEW_DATA_ROOT, "state_data")
# The dashboard's own data folder -- where finished files are written.
STATE_DATA = os.path.join(REPO_ROOT, "state_data")

COLUMNS_LIST = ["Facility Number", "School Type", "School District or Name",
                "Facility Address", "County", "MMR Vaccination Rate", "Age Group"]


TEXT_COLUMNS = ["School Type", "School District or Name", "Facility Address", "County"]

# The bytes 0x80-0xBF (UTF-8 continuation bytes) as cp1252 would render them.
# A mojibake lead char is only ever followed by one of these.
_MOJIBAKE_CONT = "".join(
    bytes([b]).decode("cp1252", errors="replace") for b in range(0x80, 0xC0)
).replace("�", "")
# California title-cased its own already-mangled text, lowercasing the 'Ã'/'Â'
# lead chars ('Montañez' -> 'Montaã±ez'), which breaks a plain round-trip.
_MOJIBAKE_LEAD = re.compile(f"([ãâ])([{re.escape(_MOJIBAKE_CONT)}])")


def clean_text(series):
    """Normalize a free-text column: repair mojibake, collapse whitespace, strip.

    Several states publish UTF-8 that was decoded as cp1252 somewhere upstream,
    leaving sequences like 'Ã¢â‚¬â€œ' where an en-dash belongs. Re-encoding as
    cp1252 and decoding as UTF-8 reverses that; text that never got mangled
    fails to round-trip and is returned untouched, so this is safe to apply
    everywhere (real 'São'/'Peña' survive intact).
    """
    def _one(x):
        if pd.isna(x):
            return x
        s = str(x)
        for _ in range(3):
            repaired = None
            for candidate in (s, _MOJIBAKE_LEAD.sub(
                    lambda m: m.group(1).upper() + m.group(2), s)):
                try:
                    repaired = candidate.encode("cp1252").decode("utf-8")
                    break
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
            if repaired is None or repaired == s:
                break
            s = repaired
        # A few source rows lost the original bytes before we ever saw them
        # ('Â??' / 'â??' where a dash belongs) -- no round-trip can recover those.
        s = re.sub(r"\s*[ÂâÃã]\?\?\s*", " - ", s).replace("�", "")
        return re.sub(r"\s+", " ", s).strip()
    return series.apply(_one)


def finalize(df):
    """Ensure exact schema, dtypes, and drop invalid/duplicate rows."""
    for col in COLUMNS_LIST:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COLUMNS_LIST].copy()

    # Normalize text before deduping, so rows differing only by stray whitespace
    # collapse into one instead of slipping past the duplicate check.
    for col in TEXT_COLUMNS:
        df[col] = clean_text(df[col])
    if "_disambig" in df.columns:
        df["_disambig"] = clean_text(df["_disambig"])

    df["MMR Vaccination Rate"] = pd.to_numeric(df["MMR Vaccination Rate"], errors="coerce")
    df = df.dropna(subset=["MMR Vaccination Rate"])
    df = df[df["MMR Vaccination Rate"] > 1e-6]
    # A handful of source rows have rate > 100% (data-entry artifacts in the
    # raw state file, e.g. count exceeding enrollment) -- drop them.
    df = df[df["MMR Vaccination Rate"] <= 100]
    df["MMR Vaccination Rate"] = df["MMR Vaccination Rate"].round(2)

    # A NaN in any groupby key makes the validator's groupby-based duplicate
    # check silently drop those rows, producing a false "duplicate" error.
    # A blank string round-trips back to NaN on CSV read, so use a placeholder.
    df["County"] = df["County"].replace("", np.nan).fillna("Unknown")

    # Disambiguate duplicate School+Age Group+County combos.
    dup_mask = df.duplicated(subset=["School District or Name", "Age Group", "County"], keep=False)
    if dup_mask.any() and "_disambig" in df.columns:
        df.loc[dup_mask, "School District or Name"] = (
            df.loc[dup_mask, "School District or Name"] + " (" + df.loc[dup_mask, "_disambig"].astype(str) + ")"
        )
    if "_disambig" in df.columns:
        df = df.drop(columns=["_disambig"])

    df = df.drop_duplicates(subset=["School District or Name", "Age Group", "County"])
    return df.reset_index(drop=True)


def parse_pct(series):
    """Parse strings like '96.8%', '≥95%', '≤5%', '>95', '<5' to floats."""
    def _one(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        s = s.replace("≥", "").replace("≤", "").replace(">", "").replace("<", "").replace("=", "")
        s = s.replace("%", "").strip()
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(_one)


def parse_frac_or_pct(series):
    """Parse a column that mixes decimal fractions (0.857) with pre-formatted
    threshold percent strings ('>95%') and suppression markers ('*', '**').
    Fractions are scaled to 0-100; percent strings are used as-is (stripped
    of their threshold symbol); markers become NaN and get dropped later.
    """
    def _one(x):
        if isinstance(x, str):
            s = x.strip()
            if s.endswith("%"):
                s = s.replace(">", "").replace("<", "").replace("%", "").strip()
                try:
                    return float(s)
                except ValueError:
                    return np.nan
            try:
                return float(s) * 100
            except ValueError:
                return np.nan
        if pd.isna(x):
            return np.nan
        return float(x) * 100
    return series.apply(_one)


def save(state_abbr, df):
    out_path = os.path.join(STATE_DATA, f"{state_abbr}_MMR_vax_rate.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")


# ---------------------------------------------------------------------------

def transform_az():
    path = os.path.join(NEW_DATA, "Arizona_2025-arizona-reporting-schools-coverage.xlsx")
    frames = []
    for sheet, age in [("Kindergarten", "Kindergarten"), ("6th Grade", "6th Grade")]:
        df = pd.read_excel(path, sheet_name=sheet)
        out = pd.DataFrame({
            "School District or Name": df["SCHOOL NAME"],
            "Facility Address": df["ADDRESS"],
            "County": df["COUNTY"].str.title(),
            "School Type": df["SCHOOL TYPE"].str.title(),
            "MMR Vaccination Rate": parse_pct(df["% IMMUNE MMR"]),
            "Age Group": age,
        })
        frames.append(out)
    return finalize(pd.concat(frames, ignore_index=True))


def transform_ca():
    path = os.path.join(NEW_DATA, "California_2024_25_CA_Kindergarten_Data_Letter.xlsx")
    df = pd.read_excel(path, sheet_name="Enrollment 20 or More", header=1)
    out = pd.DataFrame({
        "Facility Number": df["School Code"],
        "School Type": df["Public/Private"],
        "School District or Name": df["School Name"],
        "County": df["County"],
        "MMR Vaccination Rate": parse_pct(df["MMR‡ Percent"]),
        "Age Group": "Kindergarten",
    })
    return finalize(out)


def transform_co():
    path = os.path.join(
        NEW_DATA,
        "Colorado_CDPHE_Colorado_School_and_Childcare_Immunization_Facility_Data_"
        "(2023-24_school_year_through_2025-26_school_year).csv",
    )
    df = pd.read_csv(path, usecols=["ID", "Site_Name", "District_Name", "Facility_Type",
                                     "County", "Survey_Type", "Vaccine", "Metric",
                                     "Value_Percent", "Year"])
    sub = df[(df["Vaccine"] == "MMR") & (df["Metric"] == "Fully Immunized") & (df["Year"] == "2025/2026")]
    age_map = {"Kindergarten": "Kindergarten", "School": "All ages",
               "Child Care/Preschool": "Child Care/Preschool"}
    out = pd.DataFrame({
        "Facility Number": sub["ID"],
        "School Type": sub["Facility_Type"],
        "School District or Name": sub["Site_Name"],
        "County": sub["County"],
        "MMR Vaccination Rate": sub["Value_Percent"],
        "Age Group": sub["Survey_Type"].map(age_map),
        "_disambig": sub["District_Name"],
    })
    return finalize(out)


def transform_ct():
    frames = []
    for fname, age in [
        ("Connecticut_2025-2026_Kindergarten_Immunization_Rates_by_School_20260806.csv", "Kindergarten"),
        ("Connecticut_2025-2026_Seventh_Grade_Immunization_Rates_by_School_20260806.csv", "7th Grade"),
    ]:
        df = pd.read_csv(os.path.join(NEW_DATA, fname))
        out = pd.DataFrame({
            "School Type": df["School Type"],
            "School District or Name": df["School Name"],
            "Facility Address": df["Address"],
            "County": df["Planning Region"],
            "MMR Vaccination Rate": parse_pct(df["MMR"]),
            "Age Group": age,
        })
        frames.append(out)
    return finalize(pd.concat(frames, ignore_index=True))


def transform_in():
    path = os.path.join(NEW_DATA, "Indiana_immunization-data_school-year-2025-2026.xlsx")
    df = pd.read_excel(path)
    grade_map = {"K": "Kindergarten", "1": "1st Grade", "6": "6th Grade",
                 "7": "7th Grade", "12": "12th Grade"}
    out = pd.DataFrame({
        "School District or Name": df["School_Name"],
        "County": df["County"].str.title(),
        "MMR Vaccination Rate": pd.to_numeric(df["MMR_Rate"], errors="coerce") * 100,
        "Age Group": df["Grade"].astype(str).map(grade_map),
    })
    return finalize(out)


def load_committed(state_abbr):
    """Read the last committed version of a state CSV out of git.

    Used where this year's raw file drops a field that the previous cycle had
    (KS publishes no county at all). Reading the working-tree file instead
    would make the transform read back its own output on a re-run.
    """
    rel = f"state_data/{state_abbr}_MMR_vax_rate.csv"
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout
    return pd.read_csv(StringIO(blob))


# Kansas' private/diocesan entities have no USD number and so no NCES record.
# Each is filed under the county of its seat city, matching how the previous
# cycle handled the other four (Topeka -> Shawnee, Dodge City -> Ford, etc.).
KS_MANUAL_COUNTY = {
    "Z0031": "Sedgwick",  # Wichita Catholic Diocese
    "Z0070": "Sedgwick",  # Branches Academy, Wichita
}


def transform_ks():
    """Kansas publishes district-level rates with no county column at all.

    Counties come from ks_district_county.csv, built from the NCES Common Core
    of Data district directory (county of the district office). That is more
    reliable than carrying the previous cycle's values forward: it also covers
    districts new to this year's file, and it corrects ten districts the old
    file had filed under the wrong county.
    """
    lookup = pd.read_csv(os.path.join(NEW_DATA, "ks_district_county.csv"))
    code_to_county = dict(zip(lookup["Facility Number"], lookup["County"]))
    # Anything NCES does not list falls back to last cycle's value.
    old = load_committed("KS")
    for code, county in zip(old["Facility Number"], old["County"]):
        code_to_county.setdefault(code, county)
    code_to_county.update(KS_MANUAL_COUNTY)

    path = os.path.join(NEW_DATA, "Kansas_USDVaxChart_data.csv")
    df = pd.read_csv(path, encoding="utf-16", sep="\t")
    mmr = df[df["Vaccine"] == "MMR"]
    out = pd.DataFrame({
        "Facility Number": mmr["District"],
        "School District or Name": mmr["District Name"],
        "County": mmr["District"].map(code_to_county),
        "MMR Vaccination Rate": parse_pct(mmr["District Coverage"]),
        "Age Group": "Kindergarten",
    })
    return finalize(out)


def transform_ky():
    frames = []
    for fname, age in [
        ("Kentucky_Kindergarten Schools_data.csv", "Kindergarten"),
        ("Kentucky_Seventh Grade Schools_data.csv", "7th Grade"),
        ("Kentucky_Eleventh Schools_data.csv", "11th Grade"),
    ]:
        df = pd.read_csv(os.path.join(NEW_DATA, fname), encoding="utf-16", sep="\t")
        df.columns = [c.strip() for c in df.columns]
        mmr = df[df["Vaccination"] == "MMR"]
        out = pd.DataFrame({
            "School Type": mmr["Public/Private"],
            "School District or Name": mmr["School / Facility Name"],
            "County": mmr["County"],
            "MMR Vaccination Rate": pd.to_numeric(mmr["Pivot Field Values"], errors="coerce"),
            "Age Group": age,
        })
        frames.append(out)
    return finalize(pd.concat(frames, ignore_index=True))


def transform_la():
    path = os.path.join(NEW_DATA, "Louisiana_Download School Data.csv")
    df = pd.read_csv(path, encoding="utf-16", sep="\t")
    latest = df[df["SchoolYear"] == "2024 - 2025"]
    out = pd.DataFrame({
        "School District or Name": latest["School Name"],
        "County": latest["Parish"],
        "MMR Vaccination Rate": parse_pct(latest["MMR (>=2)"]),
        "Age Group": latest["Grade"],
    })
    return finalize(out)


def transform_me():
    path = os.path.join(NEW_DATA, "Maine_2024-25 School Vaccination Rates.xlsx")
    df = pd.read_excel(path, header=4)
    blocks = [
        ("Kindergarten", "Assessed", "2MMR"),
        ("7th Grade", "Assessed.1", "2MMR.1"),
        ("12th Grade", "Assessed.2", "2MMR.2"),
    ]
    frames = []
    for age, assessed_col, mmr_col in blocks:
        sub = df[df[assessed_col] != "-"]
        out = pd.DataFrame({
            "School District or Name": sub["School"],
            "County": sub["County"],
            "MMR Vaccination Rate": pd.to_numeric(sub[mmr_col], errors="coerce") * 100,
            "Age Group": age,
        })
        frames.append(out)
    return finalize(pd.concat(frames, ignore_index=True))


def transform_md():
    path = os.path.join(NEW_DATA, "Maryland_Percent of Kindergarteners Vaccinated by School 2025-2026_04102026.xlsx")
    df = pd.read_excel(path, sheet_name="Kindergarten", header=5)
    out = pd.DataFrame({
        # The source file labels public schools two different ways ("Public" in
        # most counties, "Public School" in a few) -- collapse to one.
        "School Type": df["Type of School"].replace({"Public School": "Public"}),
        "School District or Name": df["School Name"],
        "County": df["County"],
        "MMR Vaccination Rate": pd.to_numeric(df["% MMR"], errors="coerce"),
        "Age Group": "Kindergarten",
    })
    return finalize(out)


def transform_ma():
    frames = []
    for fname, sheet, age, mmr_col in [
        ("Massachusetts_Kindergarten Immunization Data by School 2025-2026.xlsx",
         "Kindergarten Rates by School", "Kindergarten", "2\nMMR "),
        ("Massachusetts_Grade 7 Immunization Data by School 2025-2026.xlsx",
         "Grade 7 Rates By School", "7th Grade", "2\nMMR"),
        ("Massachusetts_Grade 12 Immunization Data by School 2025-2026.xlsx",
         "Grade 12 Rates by School", "12th Grade", "2\nMMR"),
    ]:
        df = pd.read_excel(os.path.join(NEW_DATA, fname), sheet_name=sheet)
        out = pd.DataFrame({
            "School Type": df["SCHOOL TYPE"].str.title(),
            "School District or Name": df["SCHOOL NAME"],
            "County": df["COUNTY"].str.title(),
            "MMR Vaccination Rate": pd.to_numeric(df[mmr_col], errors="coerce") * 100,
            "Age Group": age,
        })
        frames.append(out)
    return finalize(pd.concat(frames, ignore_index=True))


def transform_mn():
    frames = []
    for fname, age in [("Minnesota_kschool2526.xlsx", "Kindergarten"), ("Minnesota_7school2526.xlsx", "7th Grade")]:
        sheet = "K_School" if "kschool" in fname else "7_School"
        df = pd.read_excel(os.path.join(NEW_DATA, fname), sheet_name=sheet, header=1)
        df = df[df["School District"] != "Statewide"]
        out = pd.DataFrame({
            "School District or Name": df["School Name"],
            "County": df["County"].str.title(),
            "MMR Vaccination Rate": parse_frac_or_pct(df["MMR % Vaccinated"]),
            "Age Group": age,
        })
        frames.append(out)
    return finalize(pd.concat(frames, ignore_index=True))


def transform_ny():
    path = os.path.join(NEW_DATA, "NewYork_School_Immunization_Survey___Beginning_2019-20_School_Year_20260806.csv")
    df = pd.read_csv(path)
    latest = df[df["Report Period"] == "2024-2025"]
    # A handful of rows have every identifying field ("District", "County",
    # "Street", etc.) literally set to the string "ERROR: #N/A" -- a lookup
    # failure on NY's end. They have no usable identity, so drop them rather
    # than let them collide into one bogus row under the duplicate check.
    latest = latest[latest["County"] != "ERROR: #N/A"]
    school_name = latest["School Name"].fillna(latest["District"])
    facility_number = pd.to_numeric(latest["School ID"], errors="coerce")
    out = pd.DataFrame({
        "Facility Number": facility_number,
        "School Type": latest["Type"],
        "School District or Name": school_name,
        "County": latest["County"].str.title(),
        "MMR Vaccination Rate": parse_pct(latest["Percent Immunized Measles"]),
        "Age Group": "All Grades",
        "_disambig": latest["Street"],
    })
    return finalize(out)


def transform_nc():
    path = os.path.join(NEW_DATA, "NorthCarolina_School Specific Download Page_data.csv")
    df = pd.read_csv(path, encoding="utf-16", sep="\t")
    up_to_date = df[df["Measure Names"] == "Up to Date (%)"]
    out = pd.DataFrame({
        "School Type": up_to_date["School Type"],
        "School District or Name": up_to_date["School Name"],
        "County": up_to_date["County"],
        "MMR Vaccination Rate": pd.to_numeric(up_to_date["Measure Values"], errors="coerce") * 100,
        "Age Group": "Kindergarten",
    })
    return finalize(out)


def transform_or():
    path = os.path.join(NEW_DATA, "Oregon_SchK-12.xlsx")
    df = pd.read_excel(path)
    out = pd.DataFrame({
        "School District or Name": df["SiteName"],
        "County": df["Agency"],
        "MMR Vaccination Rate": parse_pct(df["% Vaccinated: MMR2"]),
        "Age Group": "K-12",
    })
    return finalize(out)


def transform_pa():
    path = os.path.join(NEW_DATA, "Pennsylvania_school immunization rates, 2025-2026 – Pennsylvania school immunization rates.csv")
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "School District or Name": df["School"],
        "County": df["County"],
        "MMR Vaccination Rate": parse_pct(df["MMR Percent"]),
        "Age Group": df["Grade"],
    })
    return finalize(out)


def transform_tx():
    frames = []
    for fname, age in [
        ("Texas_2025-2026-school-vaccination-coverage-by-district-and-county-kg.xlsx", "Kindergarten"),
        ("Texas_2025-2026-school-vaccination-coverage-by-district-and-county-seventh-grade.xlsx", "7th Grade"),
    ]:
        df = pd.read_excel(os.path.join(NEW_DATA, fname), header=2)
        out = pd.DataFrame({
            "Facility Number": df["Facility Number"],
            # The kindergarten file says "Public School" where the 7th-grade
            # file says "Public ISD" for the same districts -- use one label.
            "School Type": df["School Type"].replace({"Public ISD": "Public School"}),
            "School District or Name": df["Facility Name"],
            "Facility Address": df["Facility Address"],
            "County": df["County"],
            "MMR Vaccination Rate": pd.to_numeric(df["MMR"], errors="coerce") * 100,
            "Age Group": age,
        })
        frames.append(out)
    return finalize(pd.concat(frames, ignore_index=True))


def transform_wa():
    path = os.path.join(NEW_DATA, "Washington_3481175-2025-2026BuildingRates.xlsx")
    df = pd.read_excel(path, header=2)
    sub = df[(df["Grade"] == "K-12") & (df["Disease or Vaccine"] == "Measles") &
             (df["Immunization Status"] == "Complete") & (df["Geography"] == "Building")]
    out = pd.DataFrame({
        "School District or Name": sub["School Name"],
        "Facility Address": sub["Address"],
        "County": sub["County"],
        "School Type": sub["School Type"].str.title(),
        "MMR Vaccination Rate": pd.to_numeric(sub["Percent"], errors="coerce") * 100,
        "Age Group": "K-12",
    })
    return finalize(out)


def transform_wi():
    path = os.path.join(NEW_DATA, "Wisconsin_p01892.xlsx")
    df = pd.read_excel(path, sheet_name="By School")
    sub = df[df["School Year"] == "2025–2026"]
    frames = []
    # "All Grades" capitalized to match NY, the only other state using that label.
    for age, col in [("Kindergarten", "Kindergarten: % Met Minimum MMR Requirements"),
                      ("All Grades", "% Met Minimum MMR Requirements")]:
        out = pd.DataFrame({
            "School District or Name": sub["School Name"],
            "County": sub["County"],
            "MMR Vaccination Rate": parse_pct(sub[col]),
            "Age Group": age,
        })
        frames.append(out)
    return finalize(pd.concat(frames, ignore_index=True))


TRANSFORMS = {
    "AZ": transform_az,
    "CA": transform_ca,
    "CO": transform_co,
    "CT": transform_ct,
    "IN": transform_in,
    "KS": transform_ks,
    "KY": transform_ky,
    "LA": transform_la,
    "ME": transform_me,
    "MD": transform_md,
    "MA": transform_ma,
    "MN": transform_mn,
    "NY": transform_ny,
    "NC": transform_nc,
    "OR": transform_or,
    "PA": transform_pa,
    "TX": transform_tx,
    "WA": transform_wa,
    "WI": transform_wi,
}


if __name__ == "__main__":
    states = sys.argv[1:] or list(TRANSFORMS.keys())
    for abbr in states:
        print(f"\n=== {abbr} ===")
        try:
            df = TRANSFORMS[abbr]()
        except Exception as e:
            print(f"FAILED: {e}")
            raise
        save(abbr, df)
