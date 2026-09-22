# Data Access

Raw World Bank Enterprise Survey data are not redistributed in this repository. The analysis requires the following datasets to be downloaded independently from the World Bank:

- **Morocco Enterprise Survey 2019**
  Reference: MAR_2019_ES_v01_M
  DOI: 10.48529/q5xp-jm46

- **Morocco World Bank Enterprise Survey 2023**
  Reference: MAR_2023_WBES_v01_M

- **Morocco COVID-19 follow-up Round 1**
  Reference: MAR_2020_ES-COVID19-R1_v01_M
  DOI: 10.48529/spy6-9n58

- **Morocco COVID-19 follow-up Round 2**
  Reference: MAR_2021_ES-COVID19-R2_v01_M
  DOI: 10.48529/gvht-6f61

- **Morocco COVID-19 follow-up Round 3**
  Available through the World Bank Enterprise Surveys platform.

## Obtaining the Data
Researchers can obtain these datasets by registering on the [World Bank Microdata Library](https://microdata.worldbank.org/) and agreeing to the terms of use.

Once obtained, configure the local path to the raw data files in `config/analysis_config.yaml` by modifying the `data_root` parameter.
