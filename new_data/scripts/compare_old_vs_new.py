"""Compare the pre-update (git HEAD) state_data CSVs against the freshly
transformed ones in the working tree, at the same granularity as the data
itself (one row per School District or Name + County + Age Group).

Outputs, written next to this script:
  - comparison_table.csv      : old rate, new rate, and diff for every matched row
  - comparison_scatter.png    : one old-vs-new scatter panel per state
  - comparison_histogram.png  : one panel per state, old vs new rate distribution (bars)
  - comparison_kde.png        : same comparison, smoothed KDE lines instead of bars
  - distribution_summary_long.csv : per state/dataset/bin row counts + proportions
  - distribution_summary_wide.csv : same, pivoted for side-by-side reading
"""
import io
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NEW_DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(NEW_DATA_ROOT)
STATE_DATA = os.path.join(REPO_ROOT, "state_data")
OUT_DIR = os.path.join(NEW_DATA_ROOT, "comparison")

STATES = ["AZ", "CA", "CO", "CT", "IN", "KS", "KY", "LA", "MA", "MD", "ME",
          "MN", "NC", "NY", "OR", "PA", "TX", "WA", "WI"]

KEY_COLS = ["School District or Name", "County", "Age Group"]

# Distribution-table bins: <50, 50-60, 60-70, 70-80, 80-85, then 1-point bins
# from 85-86 up through 99-100.
BIN_EDGES = [-0.0001, 50, 60, 70, 80] + list(range(85, 100)) + [100.0001]
BIN_LABELS = ["<50", "50-60", "60-70", "70-80", "80-85"] + \
    [f"{x}-{x+1}" for x in range(85, 100)]
assert len(BIN_EDGES) == len(BIN_LABELS) + 1

OLD_COLOR = "#2563eb"  # blue
NEW_COLOR = "#dc2626"  # red


def gaussian_kde(data, grid):
    """Minimal Gaussian KDE (Silverman bandwidth), no scipy dependency."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    std = data.std(ddof=1)
    bandwidth = 1.06 * std * n ** (-1 / 5) if std > 0 else 1.0
    diffs = (grid[:, None] - data[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs ** 2) / np.sqrt(2 * np.pi)
    return kernel.sum(axis=1) / (n * bandwidth)


def load_old(state_abbr):
    raw = subprocess.run(
        ["git", "show", f"HEAD:state_data/{state_abbr}_MMR_vax_rate.csv"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    ).stdout
    return pd.read_csv(io.StringIO(raw))


def load_new(state_abbr):
    return pd.read_csv(os.path.join(STATE_DATA, f"{state_abbr}_MMR_vax_rate.csv"))


def normalize_key(df):
    for col in KEY_COLS:
        # Collapse embedded newlines/repeated whitespace (some source files wrap
        # school names across lines) and case-fold for a robust join key.
        df[col] = (df[col].astype(str)
                   .str.replace(r"\s+", " ", regex=True)
                   .str.strip()
                   .str.upper())
    # "County"/"Parish" suffixes are inconsistently present across source years
    # (e.g. "ANNE ARUNDEL COUNTY" vs "ANNE ARUNDEL") -- strip for matching.
    df["County"] = df["County"].str.replace(r"\s+(COUNTY|PARISH)$", "", regex=True)
    return df


def compare_state(state_abbr):
    old = normalize_key(load_old(state_abbr))
    new = normalize_key(load_new(state_abbr))

    old_slim = old[KEY_COLS + ["MMR Vaccination Rate"]].rename(
        columns={"MMR Vaccination Rate": "Old Rate"})
    new_slim = new[KEY_COLS + ["MMR Vaccination Rate"]].rename(
        columns={"MMR Vaccination Rate": "New Rate"})

    # A school/age/county combo can (rarely) repeat within one file only if
    # the source itself had a near-duplicate; keep first to keep the merge 1:1.
    old_slim = old_slim.drop_duplicates(subset=KEY_COLS)
    new_slim = new_slim.drop_duplicates(subset=KEY_COLS)

    merged = old_slim.merge(new_slim, on=KEY_COLS, how="inner")
    merged.insert(0, "State", state_abbr)
    merged["Difference"] = (merged["New Rate"] - merged["Old Rate"]).round(2)
    return merged, len(old_slim), len(new_slim)


def bin_distribution(rates, state_abbr, dataset_label):
    """Row counts + proportions of `rates` falling in each BIN_LABELS bucket."""
    binned = pd.cut(rates, bins=BIN_EDGES, labels=BIN_LABELS, right=False)
    counts = binned.value_counts().reindex(BIN_LABELS, fill_value=0)
    out = pd.DataFrame({
        "State": state_abbr,
        "Dataset": dataset_label,
        "Bin": BIN_LABELS,
        "Count": counts.values,
    })
    out["Proportion"] = (out["Count"] / out["Count"].sum()).round(4)
    return out


def main():
    tables = []
    coverage = []
    dist_rows = []
    full_by_state = {}
    for state_abbr in STATES:
        merged, n_old, n_new = compare_state(state_abbr)
        tables.append(merged)
        coverage.append({
            "State": state_abbr,
            "Old rows": n_old,
            "New rows": n_new,
            "Matched rows": len(merged),
            "Match rate vs old (%)": round(100 * len(merged) / n_old, 1) if n_old else np.nan,
        })

        # Distributions use every row in each full file (not just matched
        # pairs) -- this is about the overall shape of rates each year, not
        # about paired schools.
        old_full = load_old(state_abbr)
        new_full = load_new(state_abbr)
        full_by_state[state_abbr] = (old_full, new_full)
        dist_rows.append(bin_distribution(old_full["MMR Vaccination Rate"], state_abbr, "Old"))
        dist_rows.append(bin_distribution(new_full["MMR Vaccination Rate"], state_abbr, "New"))

    full_table = pd.concat(tables, ignore_index=True)
    full_table = full_table.sort_values(["State", "County", "School District or Name", "Age Group"])
    table_path = os.path.join(OUT_DIR, "comparison_table.csv")
    full_table.to_csv(table_path, index=False)
    print(f"Wrote {table_path} ({len(full_table)} matched rows across {len(STATES)} states)")

    coverage_df = pd.DataFrame(coverage)
    coverage_path = os.path.join(OUT_DIR, "comparison_coverage.csv")
    coverage_df.to_csv(coverage_path, index=False)
    print(f"Wrote {coverage_path}")
    print(coverage_df.to_string(index=False))

    # --- scatter plot grid: one panel per state, old vs new ---
    n = len(STATES)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, state_abbr in enumerate(STATES):
        ax = axes[i]
        sub = full_table[full_table["State"] == state_abbr]
        ax.scatter(sub["Old Rate"], sub["New Rate"], s=8, alpha=0.35, color="#2b6cb0", edgecolors="none")
        ax.plot([0, 100], [0, 100], color="#d97706", linewidth=1, linestyle="--")
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 101)
        ax.set_title(f"{state_abbr} (n={len(sub)})", fontsize=11)
        ax.set_xlabel("Old rate (%)", fontsize=9)
        ax.set_ylabel("New rate (%)", fontsize=9)
        ax.tick_params(labelsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("MMR Vaccination Rate: Old vs New, by state", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = os.path.join(OUT_DIR, "comparison_scatter.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")

    # --- distribution summary tables ---
    dist_long = pd.concat(dist_rows, ignore_index=True)
    dist_long["Bin"] = pd.Categorical(dist_long["Bin"], categories=BIN_LABELS, ordered=True)
    dist_long = dist_long.sort_values(["State", "Dataset", "Bin"])
    long_path = os.path.join(OUT_DIR, "distribution_summary_long.csv")
    dist_long.to_csv(long_path, index=False)
    print(f"Wrote {long_path}")

    dist_wide = dist_long.pivot_table(index=["State", "Bin"], columns="Dataset",
                                       values=["Count", "Proportion"], observed=False)
    dist_wide.columns = [f"{metric} ({dataset})" for metric, dataset in dist_wide.columns]
    dist_wide = dist_wide.reset_index().sort_values(["State", "Bin"])
    wide_path = os.path.join(OUT_DIR, "distribution_summary_wide.csv")
    dist_wide.to_csv(wide_path, index=False)
    print(f"Wrote {wide_path}")

    # --- histogram grid: one panel per state, old vs new distribution ---
    hist_bins = np.arange(0, 101, 2)
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes2 = axes2.flatten()

    for i, state_abbr in enumerate(STATES):
        ax = axes2[i]
        old_full, new_full = full_by_state[state_abbr]
        ax.hist(old_full["MMR Vaccination Rate"], bins=hist_bins, density=True,
                alpha=0.45, color=OLD_COLOR, label="Old", edgecolor="none")
        ax.hist(new_full["MMR Vaccination Rate"], bins=hist_bins, density=True,
                alpha=0.45, color=NEW_COLOR, label="New", edgecolor="none")
        ax.set_xlim(0, 100)
        ax.set_title(f"{state_abbr} (old n={len(old_full)}, new n={len(new_full)})", fontsize=10)
        ax.set_xlabel("MMR rate (%)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)

    for j in range(n, len(axes2)):
        axes2[j].axis("off")

    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper right", fontsize=10)
    fig2.suptitle("MMR Vaccination Rate distribution: Old vs New, by state", fontsize=14)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    hist_path = os.path.join(OUT_DIR, "comparison_histogram.png")
    fig2.savefig(hist_path, dpi=150)
    print(f"Wrote {hist_path}")

    # --- KDE grid: same comparison, smoothed lines instead of bars ---
    grid = np.linspace(0, 100, 400)
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes3 = axes3.flatten()

    for i, state_abbr in enumerate(STATES):
        ax = axes3[i]
        old_full, new_full = full_by_state[state_abbr]
        old_rates = old_full["MMR Vaccination Rate"].dropna()
        new_rates = new_full["MMR Vaccination Rate"].dropna()
        ax.plot(grid, gaussian_kde(old_rates, grid), color=OLD_COLOR, linewidth=1.8, label="Old")
        ax.plot(grid, gaussian_kde(new_rates, grid), color=NEW_COLOR, linewidth=1.8, label="New")
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{state_abbr} (old n={len(old_rates)}, new n={len(new_rates)})", fontsize=10)
        ax.set_xlabel("MMR rate (%)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)

    for j in range(n, len(axes3)):
        axes3[j].axis("off")

    handles, labels = axes3[0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc="upper right", fontsize=10)
    fig3.suptitle("MMR Vaccination Rate distribution (KDE): Old vs New, by state", fontsize=14)
    fig3.tight_layout(rect=[0, 0, 1, 0.97])
    kde_path = os.path.join(OUT_DIR, "comparison_kde.png")
    fig3.savefig(kde_path, dpi=150)
    print(f"Wrote {kde_path}")


if __name__ == "__main__":
    main()
