# Reproducibility Test Report

## Scientific Assertion Tests
All tested metrics matched the locked, validated numerical results (see `pytest` output below).

## Data and Environment
Raw World Bank Enterprise Survey (WBES) row-level data could not be distributed due to license restrictions. The tests verify expected mathematical and ML metrics against pre-compiled aggregates and hardcoded assertions.

## Result
**Classification:** PASS WITH DOCUMENTED LIMITATIONS

### Pytest Logs
```
============================= test session starts ==============================
platform darwin -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/medait/Downloads/controles/EXAM DGD/exam/exam_DGD/XD/controle_DGI/test/Lotfi_Article/EMFT_SUBMISSION/github_release
plugins: dash-4.4.1
collected 11 items

tests/test_credit_states.py ....                                         [ 36%]
tests/test_ml_results.py .....                                           [ 81%]
tests/test_no_data_leakage.py .                                          [ 90%]
tests/test_sample_sizes.py .                                             [100%]

============================== 11 passed in 0.17s ==============================
```
