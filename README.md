# Credit-Market Participation and Financing Constraints among Moroccan Firms: Firm-Level Transitions and Temporal Predictability

## Authors
1. Younes Nadir (Corresponding author - younes.nadir@univh2c.ma)
2. Mohamed Rachdi
3. Lotfi Said

## Repository Purpose
This is the official reproducibility repository for the manuscript: *Credit-Market Participation and Financing Constraints among Moroccan Firms: Firm-Level Transitions and Temporal Predictability*. It contains the source code, configurations, aggregate results, and testing suites necessary to understand and independently verify the analysis pipelines used in the study.

## Data Access Constraints
**Raw World Bank Enterprise Survey data are not redistributed in this repository.**

Due to the Terms of Use of the World Bank Microdata Library, we cannot publicly distribute the row-level data files required to run this code end-to-end. Please refer to [DATA_ACCESS.md](DATA_ACCESS.md) for detailed instructions on acquiring the exact datasets used.

## Installation & Environment
Refer to `requirements.txt` or `environment.yml` to set up the Python environment:
```bash
pip install -r requirements.txt
# OR
conda env create -f environment.yml
conda activate emft-reproducibility
```

## Reproduction Instructions
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for comprehensive step-by-step documentation on pipeline execution, random seeds, and machine learning structures.

## Expected Outputs
The pipeline generates:
- Cross-sectional models and transition matrices
- Nested CV and external hold-out evaluation metrics
- Aggregated CSV files (provided in `outputs/tables/`)
- Result figures (provided in `outputs/figures/`)

## Citation
Please see `CITATION.cff` or cite the published manuscript (once available).

## License
The code in this repository is licensed under the MIT License (see `LICENSE`). Note that this license applies **only** to the repository code and does not grant any rights to the World Bank Enterprise Survey data.

## Contact
For questions regarding the methodology or codebase, please contact the corresponding author, Younes Nadir (younes.nadir@univh2c.ma).
