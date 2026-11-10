# DS2500 Admissions & Graduation Rate Analysis

Analysis of racial gaps in six-year graduation rates across U.S. four-year institutions, using IPEDS 2024 data and selectivity-based comparisons.


## Project Overview

This project examines racial disparities in six-year graduation rates across U.S. four-year colleges and universities. Using IPEDS 2024 survey files (HD, ADM, EF, GR), we analyze how graduation outcomes differ for White, Black, Hispanic, and Asian students, and how these gaps change across institutional selectivity tiers (Highly Selective, Selective, Open/Less Selective).

## Data & Methods

All datasets were drawn from IPEDS 2024 and merged using the UNITID institutional identifier. Key data sources include:

- **HD** — institutional characteristics  
- **ADM** — admissions counts, acceptance rate, yield  
- **EF** — fall enrollment by race and gender  
- **GR** — graduation cohort denominators and completers  

Graduation rates were computed at the group level (completers within 150% of normal time / adjusted cohort, the same denominator IPEDS uses for its official rate). Rates were scaled to [0, 1], zero denominators treated as missing, and flag columns removed. Institutions were categorized by selectivity using acceptance rate:

- Highly Selective: < 30%  
- Selective: 30–60%  
- Open/Less Selective: > 60%  

Gender gaps and race enrollment proportions were also calculated.

## Key Findings

- Asian students graduate at the highest rate (58.9%), followed by White (56.1%), Hispanic (48.7%), and Black students (40.9%).
- Graduation rates improve at more selective institutions for all groups, reaching the mid‑80% range at highly selective schools.
- Racial gaps narrow at highly selective institutions (e.g., Black–White gap ≈ 4 points) and widen substantially at open-access schools (≈ 17 points).
- Asian students outperform White students at every selectivity tier.
- Open-access institutions show the widest range of outcomes, from near-zero to near-perfect graduation rates.

## How to Run

Requires Python 3 (tested with 3.14).

```bash
python -m pip install -r requirements.txt
python code/cleaning.py            # merges raw IPEDS files -> data/analysis_clean.csv, data/four_year_clean.csv
python code/analysis_upgraded.py   # plots -> plots/, t-test results -> data/ttest_results.csv
```

Run `cleaning.py` first; `analysis_upgraded.py` reads its output. The scripts can be run from any directory.

## Known Limitations

- Group averages are unweighted means of school-level rates, so schools with very small subgroup cohorts (e.g., 1–5 students of a given race) count as much as large schools.
- The regression features are on different scales (enrollment is a raw count), so coefficient sizes are not directly comparable across features.

## Repository Structure

- **code/** — `cleaning.py` (merge and clean) and `analysis_upgraded.py` (analysis, statistics, plots)  
- **data/** — raw IPEDS files (`hd2024.csv`, `adm2024.csv`, `ef2024a.csv`, `gr2024.csv`) and cleaned datasets  
- **plots/** — visualizations generated from analysis  
- **Project Report - Emitis, Aesha, Isha DS 2500.pdf** — full DS2500 report

## Authors

- Isha Hassan (Lead for code and analysis)
- Emitis Dastmalchi (Report scribe) 
- Aesha Patel (Presentation developer)

## Full Report

The complete DS2500 project report is included in this repository as **Project Report - Emitis, Aesha, Isha DS 2500.pdf**.

**Note:** The report reflects our original class submission (April 2026). After the class, I retested the code and corrected the graduation-rate calculation to use IPEDS's official adjusted cohort, and fixed how schools at exactly 30% acceptance were assigned to tiers. The numbers in Key Findings above are the corrected ones. The changes are small (under 1 percentage point) and don't change any conclusions.

