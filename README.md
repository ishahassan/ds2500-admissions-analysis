# DS2500 Admissions Analysis

Data analysis project exploring demographic differences in admissions outcomes using statistical tests and visualization.

## Project description

This project is based on a DS2500 course project completed by a team of three students. It analyzes how admissions outcomes differ across racial groups and levels of institutional selectivity (high vs. open admissions). Using cleaned datasets, statistical tests, and visualizations, the project highlights where gaps are largest and how they relate to educational equity.

## Repository structure

- **code/** – Python scripts and notebooks used for cleaning and analysis  
- **data/** – Raw and cleaned datasets used in the project  
- **plots/** – Visualizations generated from the analysis  
- **analysis.zip** – Archived version of the original project files

## Data and methods

The datasets include admissions outcomes and demographic breakdowns across institutions, with variables such as race, selectivity level, and acceptance measures. Key statistical outputs include:

- Mean scores for high vs. open admissions institutions
- t-statistics and p-values comparing groups
- Cohen’s d effect sizes

Methods used:

- Data cleaning with Python (pandas)
- Independent samples t-tests
- Effect size calculation (Cohen’s d)
- Visualization with matplotlib and seaborn

## Key findings

- Differences between high and open admissions institutions are statistically significant across all racial groups (very small p-values).
- Effect sizes are large, indicating meaningful practical differences in outcomes.
- Black and Asian groups show some of the largest gaps between high and open admissions outcomes.
- Visualizations reveal clear patterns in selectivity and acceptance that align with concerns about educational equity.
