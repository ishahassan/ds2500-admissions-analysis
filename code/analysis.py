"""
DS 2500 - IPEDS Data Project
analysis.py — Racial Graduation Gaps at 4-Year Institutions
=============================================================
Research Question:
  At 4-year institutions, which racial/ethnic groups face the largest
  graduation gaps — and does school selectivity make it better or worse?
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

os.chdir(r"C:\Users\ishah\Documents\GitHub\ds2500\project thingsss")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("four_year_clean.csv")

# Keep only the three meaningful 4-year sectors (drop admin units)
df = df[df["sector_label"].isin([
    "Public 4-year",
    "Private nonprofit 4-year",
    "Private for-profit 4-year"
])].copy()

print(f"Working dataset: {len(df):,} four-year schools")
print(df["sector_label"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: NATIONAL BASELINE — Average grad rate by racial group
# ─────────────────────────────────────────────────────────────────────────────

race_cols = {
    "grad_rate_white"    : "White",
    "grad_rate_asian"    : "Asian",
    "grad_rate_hispanic" : "Hispanic",
    "grad_rate_black"    : "Black",
}

# Compute national averages (only schools that have data for each group)
nat_avg = {label: df[col].mean() for col, label in race_cols.items()}
nat_avg["Overall"] = df["grad_rate"].mean()

nat_df = pd.Series(nat_avg).sort_values(ascending=False)
print("\nNational avg 6-year grad rates:")
print(nat_df.apply(lambda x: f"{x:.1%}").to_string())

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2ecc71" if g == "Overall" else
          "#3498db" if g == "White" else
          "#9b59b6" if g == "Asian" else
          "#e67e22" if g == "Hispanic" else
          "#e74c3c"
          for g in nat_df.index]
bars = ax.bar(nat_df.index, nat_df.values, color=colors, edgecolor="white", width=0.6)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, 1)
ax.set_title("Average 6-Year Graduation Rate by Race/Ethnicity\nAll 4-Year Institutions (2024)", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Avg 6-Year Graduation Rate")
for bar, val in zip(bars, nat_df.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.1%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("plot1_national_baseline.png", dpi=150)
plt.close()
print("Saved plot1_national_baseline.png")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: SELECTIVITY BUCKETS
# Create 3 groups based on acceptance rate
# ─────────────────────────────────────────────────────────────────────────────

# Only schools that report admissions data
df_adm = df.dropna(subset=["accept_rate", "grad_rate"]).copy()

df_adm["selectivity"] = pd.cut(
    df_adm["accept_rate"],
    bins=[0, 0.30, 0.60, 1.01],
    labels=["Highly Selective\n(< 30%)", "Selective\n(30–60%)", "Open/Less Selective\n(> 60%)"]
)

print("Selectivity bucket counts:")
print(df_adm["selectivity"].value_counts().sort_index().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: GRAD RATES BY RACE WITHIN EACH SELECTIVITY BUCKET
# ─────────────────────────────────────────────────────────────────────────────

sel_groups = df_adm.groupby("selectivity", observed=True)[list(race_cols.keys())].mean()
sel_groups = sel_groups.rename(columns=race_cols)

print("\nGrad rates by race and selectivity:")
print(sel_groups.map(lambda x: f"{x:.1%}").to_string())

# Grouped bar chart
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(sel_groups))
width = 0.18
group_colors = {"White": "#3498db", "Asian": "#9b59b6",
                "Hispanic": "#e67e22", "Black": "#e74c3c"}

for i, (group, color) in enumerate(group_colors.items()):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, sel_groups[group], width, label=group,
                  color=color, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(sel_groups.index, fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, 1)
ax.set_title("6-Year Graduation Rates by Race/Ethnicity and School Selectivity\n4-Year Institutions (2024)", fontsize=13)
ax.set_ylabel("Avg 6-Year Graduation Rate")
ax.set_xlabel("School Selectivity (Acceptance Rate)")
ax.legend(title="Race/Ethnicity", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("plot2_selectivity_by_race.png", dpi=150)
plt.close()
print("Saved plot2_selectivity_by_race.png")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: GAP RELATIVE TO WHITE STUDENTS BY SELECTIVITY
# How much does the gap shrhink (or grow) as schools get more selective?
# ─────────────────────────────────────────────────────────────────────────────

gap_df = sel_groups.copy()
gap_df["Black gap"]    = gap_df["White"] - gap_df["Black"]
gap_df["Hispanic gap"] = gap_df["White"] - gap_df["Hispanic"]
gap_df["Asian gap"]    = gap_df["White"] - gap_df["Asian"]

print("\nGap vs. White students by selectivity (positive = lower than White):")
print(gap_df[["Black gap","Hispanic gap","Asian gap"]].map(lambda x: f"{x:+.1%}").to_string())

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(gap_df))
width = 0.25
ax.bar(x - width, gap_df["Black gap"],    width, label="Black vs. White",    color="#e74c3c", edgecolor="white")
ax.bar(x,         gap_df["Hispanic gap"], width, label="Hispanic vs. White",  color="#e67e22", edgecolor="white")
ax.bar(x + width, gap_df["Asian gap"],    width, label="Asian vs. White",     color="#9b59b6", edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(gap_df.index, fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Graduation Rate Gap Relative to White Students\nBy School Selectivity (2024)", fontsize=13)
ax.set_ylabel("Gap vs. White Students\n(positive = lower grad rate)")
ax.set_xlabel("School Selectivity (Acceptance Rate)")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("plot3_gap_by_selectivity.png", dpi=150)
plt.close()
print("Saved plot3_gap_by_selectivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: SCATTER — Acceptance Rate vs Grad Rate colored by sector
# ─────────────────────────────────────────────────────────────────────────────

sector_colors = {
    "Public 4-year"            : "#3498db",
    "Private nonprofit 4-year" : "#2ecc71",
    "Private for-profit 4-year": "#e74c3c",
}

fig, ax = plt.subplots(figsize=(9, 6))
for sector, color in sector_colors.items():
    sub = df_adm[df_adm["sector_label"] == sector]
    ax.scatter(sub["accept_rate"], sub["grad_rate"],
               alpha=0.35, s=18, color=color, label=sector)

ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Acceptance Rate vs. 6-Year Graduation Rate\nBy School Sector (2024)", fontsize=13)
ax.set_xlabel("Acceptance Rate (lower = more selective)")
ax.set_ylabel("6-Year Graduation Rate")
ax.legend(title="Sector", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("plot4_scatter_accept_vs_grad.png", dpi=150)
plt.close()
print("Saved plot4_scatter_accept_vs_grad.png")

print("\n All plots saved!")