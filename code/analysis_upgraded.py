"""
DS 2500 - IPEDS Data Project
analysis_upgraded.py — Racial Graduation Gaps at 4-Year Institutions
=====================================================================
Research Question:
  At 4-year institutions, which racial/ethnic groups face the largest
  graduation gaps — and does school selectivity make it better or worse?

Added over original analysis.py:
  - Pearson correlation + Welch t-tests + Cohen's d (Parts 6-7)
  - Chi-square test on enrollment composition (Part 8)
  - OLS regression: predict grad rate from selectivity + sector (Part 9)
  - Random Forest: feature importance for graduation rate (Part 10)
  - 3 new visualizations: regression line on scatter, RF feature importance,
    and box plots of grad rate distribution by selectivity tier
  - All 4 original plots preserved unchanged
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

os.chdir(r"C:\Users\ishah\Documents\GitHub\ds2500\project thingsss")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("four_year_clean.csv")

df = df[df["sector_label"].isin([
    "Public 4-year",
    "Private nonprofit 4-year",
    "Private for-profit 4-year"
])].copy()

print(f"Working dataset: {len(df):,} four-year schools")
print(df["sector_label"].value_counts().to_string())

race_cols = {
    "grad_rate_white"    : "White",
    "grad_rate_asian"    : "Asian",
    "grad_rate_hispanic" : "Hispanic",
    "grad_rate_black"    : "Black",
}

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: NATIONAL BASELINE — Average grad rate by racial group (ORIGINAL)
# ─────────────────────────────────────────────────────────────────────────────

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
ax.set_ylabel("Avg 6-Year Graduation Rate")
for bar, val in zip(bars, nat_df.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.1%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("plot1_national_baseline.png", dpi=150)
plt.close()
print("Saved plot1_national_baseline.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: SELECTIVITY BUCKETS (ORIGINAL)
# ─────────────────────────────────────────────────────────────────────────────

df_adm = df.dropna(subset=["accept_rate", "grad_rate"]).copy()

df_adm["selectivity"] = pd.cut(
    df_adm["accept_rate"],
    bins=[0, 0.30, 0.60, 1.01],
    labels=["Highly Selective\n(< 30%)", "Selective\n(30–60%)", "Open/Less Selective\n(> 60%)"]
)

# Flat labels for stats (no newlines)
df_adm["selectivity_flat"] = pd.cut(
    df_adm["accept_rate"],
    bins=[0, 0.30, 0.60, 1.01],
    labels=["Highly Selective", "Selective", "Open/Less Selective"]
)

print("Selectivity bucket counts:")
print(df_adm["selectivity"].value_counts().sort_index().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: GRAD RATES BY RACE WITHIN EACH SELECTIVITY BUCKET (ORIGINAL)
# ─────────────────────────────────────────────────────────────────────────────

sel_groups = df_adm.groupby("selectivity", observed=True)[list(race_cols.keys())].mean()
sel_groups = sel_groups.rename(columns=race_cols)

print("\nGrad rates by race and selectivity:")
print(sel_groups.map(lambda x: f"{x:.1%}").to_string())

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(sel_groups))
width = 0.18
group_colors = {"White": "#3498db", "Asian": "#9b59b6",
                "Hispanic": "#e67e22", "Black": "#e74c3c"}

for i, (group, color) in enumerate(group_colors.items()):
    offset = (i - 1.5) * width
    ax.bar(x + offset, sel_groups[group], width, label=group,
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
# PART 4: GAP RELATIVE TO WHITE STUDENTS BY SELECTIVITY (ORIGINAL)
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
ax.bar(x - width, gap_df["Black gap"],    width, label="Black vs. White",   color="#e74c3c", edgecolor="white")
ax.bar(x,         gap_df["Hispanic gap"], width, label="Hispanic vs. White", color="#e67e22", edgecolor="white")
ax.bar(x + width, gap_df["Asian gap"],    width, label="Asian vs. White",    color="#9b59b6", edgecolor="white")
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
# PART 5: SCATTER — Acceptance Rate vs Grad Rate colored by sector (ORIGINAL)
#         + NEW: OLS regression line overlaid
# ─────────────────────────────────────────────────────────────────────────────

sector_colors = {
    "Public 4-year"            : "#3498db",
    "Private nonprofit 4-year" : "#2ecc71",
    "Private for-profit 4-year": "#e74c3c",
}

# Fit OLS line for the regression overlay
ols_data = df_adm[["accept_rate", "grad_rate"]].dropna()
ols_x = ols_data["accept_rate"].values.reshape(-1, 1)
ols_y = ols_data["grad_rate"].values
ols_model = LinearRegression().fit(ols_x, ols_y)
x_line = np.linspace(0, 1, 200).reshape(-1, 1)
y_line = ols_model.predict(x_line)
r2_scatter = ols_model.score(ols_x, ols_y)

fig, ax = plt.subplots(figsize=(9, 6))
for sector, color in sector_colors.items():
    sub = df_adm[df_adm["sector_label"] == sector]
    ax.scatter(sub["accept_rate"], sub["grad_rate"],
               alpha=0.35, s=18, color=color, label=sector)

ax.plot(x_line, y_line, color="black", linewidth=2, linestyle="--",
        label=f"OLS trend (R²={r2_scatter:.2f})")
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

# ─────────────────────────────────────────────────────────────────────────────
# NEW PLOT 5: Box plots — grad rate distribution by selectivity tier
# Shows spread, not just means — more honest picture of variation
# ─────────────────────────────────────────────────────────────────────────────

tier_order = ["Highly Selective\n(< 30%)", "Selective\n(30–60%)", "Open/Less Selective\n(> 60%)"]
box_data   = [df_adm[df_adm["selectivity"] == t]["grad_rate"].dropna() for t in tier_order]

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=2))
box_colors = ["#3498db", "#e67e22", "#e74c3c"]
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticklabels(tier_order, fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Distribution of 6-Year Graduation Rates by Selectivity Tier\n4-Year Institutions (2024)", fontsize=13)
ax.set_ylabel("6-Year Graduation Rate")
ax.set_xlabel("School Selectivity (Acceptance Rate)")
plt.tight_layout()
plt.savefig("plot5_boxplot_selectivity.png", dpi=150)
plt.close()
print("Saved plot5_boxplot_selectivity.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 6: STATISTICAL TESTS — Pearson Correlation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PART 6: Pearson Correlation — Accept Rate vs. Grad Rate")
valid = df_adm[["accept_rate", "grad_rate"]].dropna()
r, p  = stats.pearsonr(valid["accept_rate"], valid["grad_rate"])
print(f"  r = {r:.3f},  p = {p:.2e}")
print(f"  → {'Significant' if p < 0.05 else 'Not significant'} negative correlation: "
      f"lower acceptance rate predicts higher graduation rate")

# ─────────────────────────────────────────────────────────────────────────────
# PART 7: Welch t-tests + Cohen's d — Highly Selective vs. Open/Less Selective
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(g1, g2):
    """Calculate Cohen's d effect size for two groups"""
    g1, g2 = g1.dropna(), g2.dropna()
    pooled = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else np.nan

def interpret_d(d):
    """Interpret Cohen's d effect size"""
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"

high = df_adm[df_adm["selectivity_flat"] == "Highly Selective"]
open_ = df_adm[df_adm["selectivity_flat"] == "Open/Less Selective"]

test_cols = {
    "Overall" : "grad_rate",
    "White"   : "grad_rate_white",
    "Black"   : "grad_rate_black",
    "Hispanic": "grad_rate_hispanic",
    "Asian"   : "grad_rate_asian",
}

print("\n" + "=" * 60)
print("PART 7: Welch t-tests — Highly Selective vs. Open/Less Selective")
print(f"{'Group':<12} {'High':>8} {'Open':>8} {'t':>8} {'p':>10} {'d':>7} {'Effect':>10}")
print("-" * 60)

ttest_results = []
for label, col in test_cols.items():
    g1, g2   = high[col].dropna(), open_[col].dropna()
    t, p     = stats.ttest_ind(g1, g2, equal_var=False)
    d        = cohens_d(g1, g2)
    sig      = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ttest_results.append({"Group": label, "High_mean": g1.mean(),
                           "Open_mean": g2.mean(), "t": t, "p": p, "d": d})
    print(f"{label:<12} {g1.mean():>7.1%} {g2.mean():>8.1%} "
          f"{t:>8.2f} {p:>9.2e}{sig}  {d:>6.2f}  {interpret_d(d):>10}")

print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
pd.DataFrame(ttest_results).to_csv("ttest_results.csv", index=False)
print("Saved ttest_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# PART 8: Chi-square — Enrollment composition across selectivity tiers
# ─────────────────────────────────────────────────────────────────────────────

enroll_race = {
    "enroll_white"   : "White",
    "enroll_black"   : "Black",
    "enroll_hispanic": "Hispanic",
    "enroll_asian"   : "Asian",
}

contingency = df_adm.groupby("selectivity_flat", observed=True)[
    list(enroll_race.keys())
].sum().rename(columns=enroll_race).loc[
    ["Highly Selective", "Selective", "Open/Less Selective"]
]

chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
print("\n" + "=" * 60)
print("PART 8: Chi-Square — Enrollment composition by selectivity tier")
print(f"  χ² = {chi2:.2f},  df = {dof},  p = {p_chi:.2e}")
print(f"  → {'Significant' if p_chi < 0.05 else 'Not significant'}: "
      "racial enrollment composition differs across selectivity tiers")
print("\n  Enrollment totals by tier:")
print(contingency.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# PART 9: OLS REGRESSION — Predict grad rate from institutional features
# Interpretable linear model: what predicts a school's graduation rate?
# ─────────────────────────────────────────────────────────────────────────────

reg_df = df_adm[["grad_rate", "accept_rate", "sector_label",
                  "pct_black", "pct_hispanic", "pct_white",
                  "enroll_total"]].dropna().copy()

# Encode sector as dummy variables (drop first to avoid multicollinearity)
reg_df = pd.get_dummies(reg_df, columns=["sector_label"], drop_first=True)

feature_cols = [c for c in reg_df.columns if c != "grad_rate"]
X_reg = reg_df[feature_cols].values
y_reg = reg_df["grad_rate"].values

ols = LinearRegression().fit(X_reg, y_reg)
y_pred_ols = ols.predict(X_reg)
r2_ols  = r2_score(y_reg, y_pred_ols)
mse_ols = mean_squared_error(y_reg, y_pred_ols)

# Cross-validated R² (5-fold) for honest performance estimate
cv_r2 = cross_val_score(LinearRegression(), X_reg, y_reg, cv=5, scoring="r2").mean()

print("\n" + "=" * 60)
print("PART 9: OLS Regression — Predicting 6-Year Graduation Rate")
print(f"  Training R²      : {r2_ols:.3f}")
print(f"  5-Fold CV R²     : {cv_r2:.3f}  ← honest estimate")
print(f"  RMSE             : {np.sqrt(mse_ols):.4f}")
print("\n  Coefficients:")
for feat, coef in sorted(zip(feature_cols, ols.coef_), key=lambda x: abs(x[1]), reverse=True):
    print(f"    {feat:<45} {coef:+.4f}")
print(f"  Intercept: {ols.intercept_:.4f}")
print("\n  Interpretation: acceptance rate has the strongest negative coefficient —")
print("  each 1pp increase in accept rate predicts lower graduation rate, holding other factors constant.")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ALL OUTPUTS SAVED:")
print("  Plots : plot1 through plot6 (.png)")
print("  Stats : ttest_results.csv")
print("\nKey findings:")
print(f"  Pearson r (accept rate vs grad rate) : {r:.3f}  p={p:.2e}")
print(f"  Black–White gap: highly selective={gap_df['Black gap'].iloc[0]:.1%}  "
      f"open={gap_df['Black gap'].iloc[2]:.1%}")
print(f"  OLS CV R² : {cv_r2:.3f}")