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

Graduation rates were computed at the group level (completers / cohort size). Rates were scaled to [0, 1], infinite values replaced with NaN, and flag columns removed. Institutions were categorized by selectivity using acceptance rate:

- Highly Selective: < 30%  
- Selective: 30–60%  
- Open/Less Selective: > 60%  

Gender gaps and race enrollment proportions were also calculated.

## Key Findings

- Asian students graduate at the highest rate (58.8%), followed by White (56.0%), Hispanic (48.6%), and Black students (40.9%).
- Graduation rates improve at more selective institutions for all groups, reaching the mid‑80% range at highly selective schools.
- Racial gaps narrow at highly selective institutions (e.g., Black–White gap ≈ 4 points) and widen substantially at open-access schools (≈ 17 points).
- Asian students outperform White students at every selectivity tier.
- Open-access institutions show the widest variation in outcomes, ranging from near-zero to near-perfect graduation rates.

## Repository Structure

- **code/** — Python scripts and notebooks  
- **data/** — raw and cleaned datasets  
- **plots/** — visualizations generated from analysis  
- **analysis.zip** — archived project files  
- **Project Report.pdf** — full DS2500 report

## Authors

- Isha Hassan (Lead for code and analysis)
- Emitis Dastmalchi (Report scribe) 
- Aesha Patel (Presentation developer)

## Full Report

The complete DS2500 project report is included in this repository as **Project Report - Emitis, Aesha, Isha DS 2500.pdf**.

