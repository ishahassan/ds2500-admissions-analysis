"""
DS 2500 - IPEDS Data Project
cleaning.py  —  Merge & Clean All Four Datasets
=================================================
"""
import os
import pandas as pd
import numpy as np

os.chdir(r"C:\Users\ishah\Documents\GitHub\ds2500\project thingsss")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD RAW FILES
# ─────────────────────────────────────────────────────────────────────────────

# adm = pd.read_csv("adm2024.csv")
# ef  = pd.read_csv("ef2024a (1).csv")
# gr  = pd.read_csv("gr2024 2.csv")
# hd  = pd.read_csv("hd2024.csv", encoding="utf-8-sig")

print("loading adm...")
adm = pd.read_csv("adm2024.csv")
print("loading ef...")
ef  = pd.read_csv("ef2024a (1).csv")
print("loading gr...")
gr  = pd.read_csv("gr2024 2.csv")
print("loading hd...")
hd  = pd.read_csv("hd2024.csv", encoding="utf-8-sig")
print("all loaded!")

print("ADM columns:", list(adm.columns[:5]))
print("EF columns:", list(ef.columns[:5]))
print("GR columns:", list(gr.columns[:5]))
print("HD columns:", list(hd.columns[:5]))

print("Loaded raw files:")
print(f"  ADM : {adm.shape[0]:,} rows x {adm.shape[1]} cols")
print(f"  EF  : {ef.shape[0]:,} rows x {ef.shape[1]} cols")
print(f"  GR  : {gr.shape[0]:,} rows x {gr.shape[1]} cols")
print(f"  HD  : {hd.shape[0]:,} rows x {hd.shape[1]} cols")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: CLEAN HD (Institutional Characteristics)
# ─────────────────────────────────────────────────────────────────────────────
# Keep only the columns useful for grouping and labeling schools.

hd_clean = hd[[
    "UNITID", "INSTNM", "STABBR", "CITY",
    "SECTOR", "CONTROL", "ICLEVEL", "INSTSIZE", "LOCALE"
]].copy()

# Decode SECTOR into a readable label
sector_map = {
    0: "Administrative Unit",
    1: "Public 4-year",
    2: "Private nonprofit 4-year",
    3: "Private for-profit 4-year",
    4: "Public 2-year",
    5: "Private nonprofit 2-year",
    6: "Private for-profit 2-year",
    7: "Public less-than-2-year",
    8: "Private nonprofit less-than-2-year",
    9: "Private for-profit less-than-2-year",
    99: "Unknown"
}

hd_clean["sector_label"] = hd_clean["SECTOR"].map(sector_map)

# Decode CONTROL
control_map = {1: "Public", 2: "Private nonprofit", 3: "Private for-profit"}
hd_clean["control_label"] = hd_clean["CONTROL"].map(control_map)

# Decode ICLEVEL (degree level focus)
iclevel_map = {1: "4-year", 2: "2-year", 3: "Less-than-2-year"}
hd_clean["level_label"] = hd_clean["ICLEVEL"].map(iclevel_map)

# Decode INSTSIZE (total enrollment size buckets)
instsize_map = {
    1: "Under 1,000",
    2: "1,000–4,999",
    3: "5,000–9,999",
    4: "10,000–19,999",
    5: "20,000 or more"
}
hd_clean["size_label"] = hd_clean["INSTSIZE"].map(instsize_map)

print(f"\nHD clean: {hd_clean.shape[0]:,} schools")
print(hd_clean["sector_label"].value_counts().head(6).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: CLEAN ADM (Admissions)
# ─────────────────────────────────────────────────────────────────────────────
# ADM is already one row per school — just need to drop the X flag columns and
# compute rates.

adm_clean = adm.drop(columns=[c for c in adm.columns if c.startswith("X")]).copy()

# Core counts
adm_clean = adm_clean.rename(columns={
    "APPLCN"  : "applied_total",
    "APPLCNM" : "applied_male",
    "APPLCNW" : "applied_female",
    "ADMSSN"  : "admitted_total",
    "ADMSSNM" : "admitted_male",
    "ADMSSNW" : "admitted_female",
    "ENRLT"   : "enrolled_total",
    "ENRLM"   : "enrolled_male",
    "ENRLW"   : "enrolled_female",
})

# Computed rates
adm_clean["accept_rate"] = (adm_clean["admitted_total"] / adm_clean["applied_total"]).clip(0, 1)
adm_clean["yield_rate"]  = (adm_clean["enrolled_total"] / adm_clean["admitted_total"]).clip(0, 1)

# SAT/ACT columns — keep the score distributions (25th/50th/75th percentiles)
score_cols = ["SATNUM", "SATPCT", "ACTNUM", "ACTPCT",
              "SATVR25", "SATVR50", "SATVR75",   # SAT Reading
              "SATMT25", "SATMT50", "SATMT75",   # SAT Math
              "ACTCM25", "ACTCM50", "ACTCM75"]   # ACT Composite
# These are already in adm_clean — no renaming needed

print(f"\nADM clean: {adm_clean.shape[0]:,} schools")
print(f"  Avg acceptance rate : {adm_clean['accept_rate'].mean():.1%}")
print(f"  Avg yield rate      : {adm_clean['yield_rate'].mean():.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: CLEAN EF (Fall Enrollment)
# ─────────────────────────────────────────────────────────────────────────────
# EF has many rows per school. Filter to:
#   EFALEVEL == 2  →  all undergraduate students
#   LINE == 99
#       →  grand total row (not broken into sub-groups)

ef_ug = ef[(ef["EFALEVEL"] == 2) & (ef["LINE"] == 99)].copy()

# Drop X flag columns
ef_ug = ef_ug.drop(columns=[c for c in ef_ug.columns if c.startswith("X")])

# Rename to readable labels (T = total, M = male, W = female)
ef_ug = ef_ug.rename(columns={
    "EFTOTLT" : "enroll_total",
    "EFTOTLM" : "enroll_male",
    "EFTOTLW" : "enroll_female",
    "EFAIANT" : "enroll_aian",       # American Indian / Alaska Native
    "EFASIAT" : "enroll_asian",
    "EFBKAAT" : "enroll_black",
    "EFHISPT" : "enroll_hispanic",
    "EFNHPIT" : "enroll_nhpi",       # Native Hawaiian / Pacific Islander
    "EFWHITT" : "enroll_white",
    "EF2MORT" : "enroll_twomore",
    "EFUNKNT" : "enroll_unknown",
    "EFNRALT" : "enroll_nonresident",
})

# Race/ethnicity proportions (share of total enrollment)
race_cols = ["enroll_aian", "enroll_asian", "enroll_black", "enroll_hispanic",
             "enroll_nhpi", "enroll_white", "enroll_twomore",
             "enroll_unknown", "enroll_nonresident"]

for col in race_cols:
    pct_col = col.replace("enroll_", "pct_")
    ef_ug[pct_col] = (ef_ug[col] / ef_ug["enroll_total"]).clip(0, 1)

ef_ug["pct_female"] = (ef_ug["enroll_female"] / ef_ug["enroll_total"]).clip(0, 1)

# Keep only the useful columns
ef_keep = ["UNITID", "enroll_total", "enroll_male", "enroll_female", "pct_female"] + \
          race_cols + [c for c in ef_ug.columns if c.startswith("pct_")]
ef_clean = ef_ug[[c for c in ef_keep if c in ef_ug.columns]].copy()

print(f"\nEF clean: {ef_clean.shape[0]:,} schools (undergrad totals)")
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: CLEAN GR (Graduation Rates)
# ─────────────────────────────────────────────────────────────────────────────
# GRTYPE=1, CHRTSTAT=10 → total cohort (denominator)
# GRTYPE=3, CHRTSTAT=13 → completers within 150% of normal time (numerator)
# (GRTYPE=2 is the cohort after exclusions, not completers)

gr = gr.drop(columns=[c for c in gr.columns if c.startswith("X")])

# Cohort size
gr_cohort = gr[(gr["GRTYPE"] == 1) & (gr["CHRTSTAT"] == 10)][
    ["UNITID", "GRTOTLT", "GRTOTLM", "GRTOTLW",
     "GRBKAAT", "GRHISPT", "GRWHITT", "GRASIAT"]
].rename(columns={
    "GRTOTLT" : "cohort_total",
    "GRTOTLM" : "cohort_male",
    "GRTOTLW" : "cohort_female",
    "GRBKAAT" : "cohort_black",
    "GRHISPT" : "cohort_hispanic",
    "GRWHITT" : "cohort_white",
    "GRASIAT" : "cohort_asian",
})

# 150% completers
gr_completers = gr[(gr["GRTYPE"] == 3) & (gr["CHRTSTAT"] == 13)][
    ["UNITID", "GRTOTLT", "GRTOTLM", "GRTOTLW",
     "GRBKAAT", "GRHISPT", "GRWHITT", "GRASIAT"]
].rename(columns={
    "GRTOTLT" : "grad_total",
    "GRTOTLM" : "grad_male",
    "GRTOTLW" : "grad_female",
    "GRBKAAT" : "grad_black",
    "GRHISPT" : "grad_hispanic",
    "GRWHITT" : "grad_white",
    "GRASIAT" : "grad_asian",
})

# Merge cohort + completers
gr_merged = gr_cohort.merge(gr_completers, on="UNITID", how="inner")

# Compute graduation rates
# Compute graduation rates — divide each group's completers by their own cohort size
gr_merged["grad_rate"]          = (gr_merged["grad_total"]    / gr_merged["cohort_total"]).clip(0, 1)
gr_merged["grad_rate_male"]     = (gr_merged["grad_male"]     / gr_merged["cohort_male"]).clip(0, 1)
gr_merged["grad_rate_female"]   = (gr_merged["grad_female"]   / gr_merged["cohort_female"]).clip(0, 1)
gr_merged["grad_rate_black"]    = (gr_merged["grad_black"]    / gr_merged["cohort_black"]).clip(0, 1)
gr_merged["grad_rate_hispanic"] = (gr_merged["grad_hispanic"] / gr_merged["cohort_hispanic"]).clip(0, 1)
gr_merged["grad_rate_white"]    = (gr_merged["grad_white"]    / gr_merged["cohort_white"]).clip(0, 1)
gr_merged["grad_rate_asian"]    = (gr_merged["grad_asian"]    / gr_merged["cohort_asian"]).clip(0, 1)

# Gender gap: positive = women graduate at higher rate
gr_merged["gender_gap"] = gr_merged["grad_rate_female"] - gr_merged["grad_rate_male"]

gr_merged = gr_merged.replace([np.inf, -np.inf], np.nan)

print(f"\nGR clean: {gr_merged.shape[0]:,} schools (bachelor's seekers)")
print(f"  Avg 6-year grad rate : {gr_merged['grad_rate'].mean():.1%}")
print(f"  Avg gender gap       : {gr_merged['gender_gap'].mean():+.1%}  (+ = women higher)")
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: MERGE INTO ONE ANALYSIS DATASET
# ─────────────────────────────────────────────────────────────────────────────
# UNITID is the shared key across all IPEDS files.
# Start with HD (all schools), then left-join the others so we keep
# every institution and just get NaN where data is missing.

analysis = hd_clean.merge(adm_clean, on="UNITID", how="left")
analysis = analysis.merge(ef_clean,  on="UNITID", how="left")
analysis = analysis.merge(
    gr_merged[["UNITID", "cohort_total", "grad_rate", "grad_rate_male",
               "grad_rate_female", "grad_rate_black", "grad_rate_hispanic",
               "grad_rate_white", "grad_rate_asian", "gender_gap"]],
    on="UNITID", how="left"
)

print(f"\n FINAL analysis dataset: {analysis.shape[0]:,} rows x {analysis.shape[1]} cols")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: MISSING DATA SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- % Missing by column (only cols with any missing) ---")
missing = (analysis.isnull().mean() * 100).round(1)
print(missing[missing > 0].sort_values(ascending=False).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: SAVE CLEAN FILE
# ─────────────────────────────────────────────────────────────────────────────

analysis.to_csv("analysis_clean.csv", index=False)
print("\n SAVED to analysis_clean.csv — use for analysis!")

# Filter to 4-year schools only for analysis
four_year = analysis[analysis["level_label"] == "4-year"].copy()
four_year.to_csv("four_year_clean.csv", index=False)
print(f"\n 4-year schools only: {four_year.shape[0]:,} schools")
print(f"   With grad rate data : {four_year['grad_rate'].notna().sum():,} schools")
print(f"   With admissions data: {four_year['accept_rate'].notna().sum():,} schools")

# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK — preview key columns for a few schools
# ─────────────────────────────────────────────────────────────────────────────

preview_cols = ["INSTNM", "STABBR", "sector_label",
                "accept_rate", "enroll_total", "grad_rate", "gender_gap"]
print("\n--- Sample rows (4-year schools with full data) ---")
sample = analysis[analysis["level_label"] == "4-year"].dropna(
    subset=["accept_rate", "grad_rate"]
)[preview_cols].head(10)
print(sample.to_string(index=False))