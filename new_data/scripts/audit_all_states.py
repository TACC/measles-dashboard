"""Full-sweep QA audit of every state CSV in state_data/.

Goes beyond vax_rate_csv_checker.py: looks for formatting inconsistencies within
and across states, suspicious name/value entries, age-group drift between the
old (git HEAD) and new versions of each file, and anything else that suggests a
transform slipped.

Usage:
    python3 new_data/scripts/audit_all_states.py

Read-only -- reports findings, changes nothing.
"""

import os
import subprocess
import sys
from io import StringIO

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STATE_DATA = os.path.join(REPO, "state_data")
OUT_DIR = os.path.join(os.path.dirname(HERE), "comparison")

# The 19 states refreshed this cycle. AL/IA/ID/MI/NM were not touched.
UPDATED = ["AZ", "CA", "CO", "CT", "IN", "KS", "KY", "LA", "MA", "MD",
           "ME", "MN", "NC", "NY", "OR", "PA", "TX", "WA", "WI"]

COLUMNS_LIST = ["Facility Number", "School Type", "School District or Name",
                "Facility Address", "County", "MMR Vaccination Rate", "Age Group"]

# Strings that should never survive a transform into a name/county field.
JUNK_PATTERNS = [
    (r"#N/?A|#REF|#VALUE|#DIV/0|\bERROR\b", "spreadsheet error literal"),
    (r"Ã|â€| Â|�", "mojibake / double-encoded text"),
    (r"^\s*(nan|none|null|n/?a|unknown|-+|\.+)\s*$", "placeholder-only value"),
    (r"^\s*\d+(\.\d+)?\s*$", "purely numeric name"),
    (r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "control character"),
    (r"^\s|\s$", "leading/trailing whitespace"),
    (r"\s{2,}", "collapsed-whitespace candidate"),
]

findings = []


def flag(state, severity, category, detail):
    findings.append({"State": state, "Severity": severity,
                     "Category": category, "Detail": detail})


def load_old(state):
    """Read the pre-update version of a state's CSV straight out of git HEAD."""
    rel = f"state_data/{state}_MMR_vax_rate.csv"
    try:
        blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    return pd.read_csv(StringIO(blob))


def audit_state(state):
    path = os.path.join(STATE_DATA, f"{state}_MMR_vax_rate.csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        flag(state, "HIGH", "schema", "stray index column 'Unnamed: 0' written to CSV")
        df = df.drop(columns="Unnamed: 0")

    # --- schema ---
    if list(df.columns) != COLUMNS_LIST:
        if set(df.columns) == set(COLUMNS_LIST):
            flag(state, "LOW", "schema",
                 f"columns present but out of canonical order: {list(df.columns)}")
        else:
            flag(state, "HIGH", "schema", f"column mismatch: {list(df.columns)}")

    # --- rate sanity ---
    rate = df["MMR Vaccination Rate"]
    if rate.dtype != float:
        flag(state, "HIGH", "rate", f"rate column dtype is {rate.dtype}, not float")
    if rate.isna().any():
        flag(state, "HIGH", "rate", f"{int(rate.isna().sum())} NaN rate values")
    bad = df[(rate <= 0) | (rate > 100)]
    if len(bad):
        flag(state, "HIGH", "rate", f"{len(bad)} rates outside (0, 100]")
    decimals = rate.dropna().map(
        lambda v: len(str(float(v)).split(".")[1].rstrip("0")))
    if (decimals > 2).any():
        flag(state, "MED", "rate",
             f"{int((decimals > 2).sum())} rates with >2 decimal places")
    # A cluster of rates under 1 usually means fractions leaked through unscaled.
    tiny = int((rate < 1).sum())
    if tiny > max(3, 0.01 * len(df)):
        flag(state, "HIGH", "rate",
             f"{tiny} rates below 1.0 -- possible unscaled decimal fractions")
    elif tiny:
        flag(state, "LOW", "rate",
             f"{tiny} rate(s) below 1.0 (verify these are genuine, not fractions)")

    # --- duplicates (the checker's own rule, but NaN-safe) ---
    key = ["School District or Name", "Age Group", "County"]
    dup = df[df.duplicated(subset=key, keep=False)]
    if len(dup):
        flag(state, "HIGH", "duplicates",
             f"{len(dup)} rows in {dup.groupby(key, dropna=False).ngroups} duplicate key groups")
    # NaN in a key column silently breaks the official checker's groupby.
    for col in key:
        n = int(df[col].isna().sum())
        if n:
            flag(state, "HIGH", "duplicates",
                 f"{n} NaN in key column '{col}' -- hides duplicates from the checker")

    # --- junk strings in text fields ---
    for col in ["School District or Name", "County", "School Type", "Facility Address"]:
        vals = df[col].dropna().astype(str)
        for pattern, label in JUNK_PATTERNS:
            if col == "Facility Address" and label in (
                    "purely numeric name", "collapsed-whitespace candidate"):
                continue  # addresses are legitimately numeric-ish
            hits = vals[vals.str.contains(pattern, regex=True, case=False, na=False)]
            if len(hits):
                sev = "LOW" if label in ("collapsed-whitespace candidate",
                                         "leading/trailing whitespace") else "HIGH"
                flag(state, sev, "junk-text",
                     f"{col}: {len(hits)} × {label} e.g. {hits.iloc[0]!r}")

    # --- age groups ---
    ages = sorted(df["Age Group"].dropna().unique())
    if df["Age Group"].isna().any():
        flag(state, "HIGH", "age-group", "NaN age group values")
    # Within a state, every age group should cover a comparable set of schools.
    counts = df["Age Group"].value_counts()
    if len(counts) > 1 and counts.min() < 0.2 * counts.max():
        flag(state, "MED", "age-group",
             f"very uneven age-group coverage: {counts.to_dict()}")

    # --- school type ---
    types = sorted(df["School Type"].dropna().unique().astype(str))

    # --- old vs new drift ---
    old = load_old(state)
    old_ages, old_types = [], []
    if old is not None:
        old_ages = sorted(old["Age Group"].dropna().unique())
        old_types = sorted(old["School Type"].dropna().unique().astype(str))
        gone = set(old_ages) - set(ages)
        if gone:
            flag(state, "MED", "age-group",
                 f"age group(s) present in old file but missing now: {sorted(gone)}")
        # County vocabulary drift signals a geography or capitalization change.
        oc = set(old["County"].dropna().astype(str).str.upper().str.strip())
        nc = set(df["County"].dropna().astype(str).str.upper().str.strip())
        if oc and len(oc & nc) < 0.5 * len(oc):
            flag(state, "HIGH", "county",
                 f"county vocabulary largely changed: {len(oc & nc)}/{len(oc)} old counties still present")
        ratio = len(df) / len(old) if len(old) else float("nan")
        if ratio < 0.75 or ratio > 1.5:
            flag(state, "MED", "row-count",
                 f"row count {len(old)} -> {len(df)} ({ratio:.0%} of old)")

    return {
        "State": state, "Rows": len(df),
        "OldRows": len(old) if old is not None else None,
        "AgeGroups": " | ".join(map(str, ages)),
        "OldAgeGroups": " | ".join(map(str, old_ages)),
        "SchoolTypes": " | ".join(types),
        "OldSchoolTypes": " | ".join(old_types),
        "Counties": df["County"].nunique(),
        "RateMin": round(df["MMR Vaccination Rate"].min(), 2),
        "RateMed": round(df["MMR Vaccination Rate"].median(), 2),
        "RateMax": round(df["MMR Vaccination Rate"].max(), 2),
        "AddrFilled": round(100 * df["Facility Address"].notna().mean(), 1),
        "FacNumFilled": round(100 * df["Facility Number"].notna().mean(), 1),
    }


def main():
    states = sys.argv[1:] or UPDATED
    rows = [audit_state(s) for s in states]
    summary = pd.DataFrame(rows)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== PER-STATE SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== AGE GROUPS: old vs new ===")
    for r in rows:
        marker = "  " if r["AgeGroups"] == r["OldAgeGroups"] else "->"
        print(f"{marker} {r['State']}: {r['OldAgeGroups'] or '(none)'}")
        if marker == "->":
            print(f"      now: {r['AgeGroups']}")

    print("\n=== SCHOOL TYPE VOCABULARY (cross-state consistency) ===")
    for r in rows:
        print(f"  {r['State']}: {r['SchoolTypes']}")

    print("\n=== FINDINGS ===")
    if not findings:
        print("  none")
    else:
        fdf = pd.DataFrame(findings)
        order = {"HIGH": 0, "MED": 1, "LOW": 2}
        fdf = fdf.sort_values(["Severity", "State"], key=lambda s: s.map(order).fillna(9)
                              if s.name == "Severity" else s)
        print(fdf.to_string(index=False))

    summary.to_csv(os.path.join(OUT_DIR, "audit_summary.csv"), index=False)
    pd.DataFrame(findings).to_csv(os.path.join(OUT_DIR, "audit_findings.csv"), index=False)
    print(f"\nWrote audit_summary.csv and audit_findings.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
