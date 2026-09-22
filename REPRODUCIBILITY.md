# Reproducibility Details

## Versions
- **Python version**: 3.10+
- **R version**: 4.4.1 (for Firth penalized logistic regressions)
- **Key Python Packages**: pandas, scikit-learn, interpret, catboost (see `requirements.txt`)
- **Key R Packages**: logistf, survey (versions provided in the manuscript pipeline)

## Random Seeds
- Base Seed: `2026`
- Cross-Validation Folds: `2026`, `2027`, `2028`, `2029`, `2030`
- Bootstrap Replicates: 10,000

## ML Validation Strategy
- **Nested CV Structure**: 5-fold cross-validation exclusively on the 2019 baseline sample (Internal Validation).
- **External Holdout**: 2023 standalone wave (Temporal Generalization Validation).

## Inputs & Outputs
- **Expected Inputs**: Raw `.dta` files downloaded per `DATA_ACCESS.md` mapped via `config/analysis_config.yaml`.
- **Expected Outputs**: Cleaned aggregate performance tables (`outputs/tables/`) and plots (`outputs/figures/`).

## Execution Order
If data are locally available:
1. `src/data_preparation/` to format survey items.
2. `src/cross_sectional/` and `src/longitudinal/` for matched analyses.
3. `src/covid_analysis/` for construct validity checks.
4. `scripts/run_all.py` (which internally wraps `src/ml_transportability/`) to generate CV and external bootstrap validations.

## Estimated Runtime
- Logistic Regression / EBM: ~5 minutes
- CatBoost: ~15 minutes
- Bootstrap Validation: ~20 minutes depending on multicore hardware.

## Limitations
Because raw row-level data are excluded from this release for compliance, the automated scripts provided will require manual linking to legitimately acquired files to function fully. The current reproducibility test asserts mathematically against derived aggregates only.
