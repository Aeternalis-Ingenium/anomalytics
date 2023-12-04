# Changelog

<!--next-version-placeholder-->

## v0.1.0 (04/12/2023)

- First release of `anomalytics`

### Feature

- Added documentation directory and its main modules in `anomalytics/docs/*`
- Added a list of code owners in `github/CODEOWNERS`
- Added 3 jobs: build, code-quality, and test for CI in `github/workflows/*.yaml`
- Added "QQ Plot" test for visual evaluation in `anomalytics/evals/qq_plot.py`
- Added "Kolmogorov Smirnov" test for model evaluation in `anomalytics/evals/kolmogorov_smirnov.py`
- Added anomaly detection with Peaks Over Threshold method in `anomalytics/stats/peaks_over_threshold.py`
- Added abstraction layer for all time window calculation functions in `anomalytics/time_windows/time_window.py`
- Added function to calculate time windows for POT (t0, t1, t2) in `anomalytics/time_windows/pot_window.py`
- Added lazy upload functions and its abstraction layer to create Pandas Series in `anomalytics/time_series/upload.py`
- Added project setup via TOML in `./pyproject.toml`

### Tests

- Added unit tests for evals modules in `tests/test_evaluation_methods.py`
- Added unit tests for stats modules in `tests/test_peaks_over_threshold.py`
- Added unit tests for time_windows modules in `tests/test_time_windows.py`
- Added unit tests for time_series modules in `tests/test_time_series.py`
- Added unit tests for anomalytics version in `tests/test_version.py`
- Added main test config file for future use in `tests/conftest.py`
