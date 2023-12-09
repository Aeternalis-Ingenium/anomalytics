<h1 align=center><strong>Anomalytics</strong></h1>

<h3 align=center><i>Your Ultimate Anomaly Detection & Analytics Tool</i></h3>

<p align="center">
    <a href="https://app.codecov.io/gh/Aeternalis-Ingenium/anomalytics/tree/trunk" >
        <img src="https://codecov.io/gh/Aeternalis-Ingenium/anomalytics/graph/badge.svg?token=eC84pMmUz8"/>
    </a>
    <a href="https://results.pre-commit.ci/latest/github/Aeternalis-Ingenium/anomalytics/trunk">
        <img src="https://results.pre-commit.ci/badge/github/Aeternalis-Ingenium/anomalytics/trunk.svg" alt="pre-commit.ci status">
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    </a>
    <a href="https://pycqa.github.io/isort/">
        <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="Imports: isort">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/mypy-checked-blue" alt="mypy checked">
    </a>
    <a href="https://github.com/Aeternalis-Ingenium/anomalytics/actions/workflows/build.yaml">
        <img src="https://github.com/Aeternalis-Ingenium/anomalytics/actions/workflows/build.yaml/badge.svg" alt="CI - Build">
    </a>
    <a href="https://github.com/Aeternalis-Ingenium/anomalytics/actions/workflows/code-quality.yaml">
        <img src="https://github.com/Aeternalis-Ingenium/anomalytics/actions/workflows/code-quality.yaml/badge.svg" alt="CI - Code Quality">
    </a>
    <a href="https://github.com/Aeternalis-Ingenium/anomalytics/actions/workflows/test.yaml">
        <img src="https://github.com/Aeternalis-Ingenium/anomalytics/actions/workflows/test.yaml/badge.svg" alt="CI - Automated Testing">
    </a>
    <a href="https://github.com/Aeternalis-Ingenium/anomalytics/blob/trunk/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <!-- Replace the '#' in the href with your documentation link -->
    <a href="#">
        <img src="https://img.shields.io/badge/docs-passing-brightgreen.svg" alt="Documentation">
    </a>
    <!-- Replace the '#' in the href with your PyPi package link -->
    <a href="#">
        <img src="https://img.shields.io/badge/PyPi-v0.1.0-blue.svg" alt="PyPi">
    </a>
</p>

## Installation

```shell
# Install without openpyxl
$ pip3 install anomalytics

# Install with openpyxl
$ pip3 install "anomalytics[extra]"
```

## Use Case

`anomalytics` can be used to analyze anomalies in your dataset (both as `pandas.DataFrame` or `pandas.Series`). To start, let's follow along with this minimum example where we want to detect extremely high anomalies in our time series dataset.

### Standalone

1. Import `anomalytics` and initialise your time series:

    ```python
    import anomalytics as atics

    ts = atics.read_ts(
        "my_dataset.csv",
        "csv"
    )
    ts.head()
    ```
    ```shell
    Date-Time
    2008-11-03 08:00:00   -0.282
    2008-11-03 09:00:00   -0.368
    2008-11-03 10:00:00   -0.400
    2008-11-03 11:00:00   -0.320
    2008-11-03 12:00:00   -0.155
    Name: Example Dataset, dtype: float64
    ```

2. Set the time windows of t0, t1, and t2 to compute dynamic expanding period for calculating the threshold via quantile:

    ```python
    t0, t1, t2 = atics.set_time_window(ts.shape[0], "POT", "historical", t0_pct=0.7, t1_pct=0.2, t2_pct=0.1)

    print(f"T0: {t0}")
    print(f"T1: {t1}")
    print(f"T2: {t2}")
    ```
    ```shell
    T0: 70000
    T1: 20000
    T2: 10000
    ```

3. Extract exceedances and indicate that it is a `"high"` anomaly type and what's the `q`uantile:

    ```python
    exceedance_ts = atics.get_exceedance_peaks_over_threshold(ts, ts.shape[0], "high", 0.95)
    exceedance_ts.tail()
    ```
    ```shell
    Date-Time
    2020-03-31 19:00:00    0.867
    2020-03-31 20:00:00    0.867
    2020-03-31 21:00:00    0.867
    2020-03-31 22:00:00    0.867
    2020-03-31 23:00:00    0.867
    Name: Example Dataset, dtype: float64
    ```

4. Compute the anomaly score for each exceedance and initialize a params for further analysis and evaluation:

    ```python
    params = {}
    anomaly_score_ts = atics.get_anomaly_score(exceedance_ts, exceedance_ts.shape[0], params)
    anomaly_score_ts.head()
    ```
    ```shell
    Date-Time
    2016-10-29 00:00:00    0.0
    2016-10-29 01:00:00    0.0
    2016-10-29 02:00:00    0.0
    2016-10-29 03:00:00    0.0
    2016-10-29 04:00:00    0.0
    Name: Example Dataset, dtype: float64
    ...
    ```

5. Inspect the parameters (the result of genpareto fitting):

    ```python
    print(params)
    ```
    ```shell
    {0: {'datetime': Timestamp('2016-10-29 03:00:00'),
    'c': 0.0,
    'loc': 0.0,
    'scale': 0.0,
    'p_value': 0.0,
    'anomaly_score': 0.0},
    1: {'datetime': Timestamp('2016-10-29 04:00:00'),
    'c': 0.324456778899,
    'loc': 0,
    'scale': 0.19125308567629334,
    'p_value': 0.19286132173263668,
    'anomaly_score': 5.1850728337654886},
    ...}
    ```

6. Detect the extremely high anomalies:

    ```python
    anomaly_ts = atics.detect(anomaly_score_ts, t1, 0.90)
    anomaly_ts.head()
    ```
    ```shell
    Date-Time
    2019-02-09 08:00:00    False
    2019-02-09 09:00:00    False
    2019-02-09 10:00:00    False
    2019-02-09 11:00:00    False
    2019-02-09 12:00:00    False
    Name: Example Dataset, dtype: bool
    ```

7. Evaluate your analysis result with Kolmogorov Smirnov 1 sample test:

    ```python
    ks_result = atics.evals.ks_1sample(ts=exceedance_ts, stats_method="POT", fit_params=params)
    print(ks_result)
    ```
    ```shell
    {'total_nonzero_exceedances': 5028, 'stats_distance': 0.0284, 'p_value': 0.8987, 'c': 0.003566, 'loc': 0, 'scale': 0.140657}
    ```

### Detector Instance

1. Import `anomalytics` and initialise your time series:

    ```python
    import anomalytics as atics
    ts = atics.read_ts("./my_dataset.csv", "csv")
    ts.head()
    ```
    ```shell
    Date-Time
    2008-11-03 08:00:00   -0.282
    2008-11-03 09:00:00   -0.368
    2008-11-03 10:00:00   -0.400
    2008-11-03 11:00:00   -0.320
    2008-11-03 12:00:00   -0.155
    Name: Example Dataset, dtype: float64
    ```

2. Initialize the needed detector object. Each detector utilises a different statistical method to detect the anomalies. In this example, we'll use POT method and a high anomaly type. Pay attention to the time period that is directly created where the `t2` is 1 by default because "real-time" always targets the "now" period hence 1 (sec, min, hour, day, week, month, etc.):

    ```python
    pot_detector = atics.get_detector("POT", ts, "high")

    print(f"T0: {pot_detector.t0}")
    print(f"T1: {pot_detector.t1}")
    print(f"T2: {pot_detector.t2}")
    ```
    ```shell
    T0: 70000
    T1: 19999
    T2: 1
    ```

3. The purpose of having a detector object is to support dividing the time windows between `t0`, `t1`, and `t2` because this is a big factor to extract exceedances. In case users want to customize their time window, they can call the `reset_time_window()` where users can now set t2 greater than 1 as well even though that will beat the purpose of the detector object. Pay attention to the period parameters because the method expects a percentage representation of the distribution of period (ranging 0.0 to 1.0):

    ```python
    pot_detector.reset_time_window(
        "historical",
        t0_pct=0.7,
        t1_pct=0.2,
        t2_pct=0.1,
    )

    print(f"T0: {pot_detector.t0}")
    print(f"T1: {pot_detector.t1}")
    print(f"T2: {pot_detector.t2}")
    ```
    ```shell
    T0: 70000
    T1: 20000
    T2: 10000
    ```

4. Let's extract exceedances by giving the expected `q`uantile:

    ```python
    pot_detector.get_extremes(0.95)

    exceedance_ts = pot_detector.return_dataset("exceedance")
    exceedance_ts.tail()
    ```
    ```shell
    Date-Time
    2020-03-31 19:00:00    0.867
    2020-03-31 20:00:00    0.867
    2020-03-31 21:00:00    0.867
    2020-03-31 22:00:00    0.867
    2020-03-31 23:00:00    0.867
    Name: Example Dataset, dtype: float64
    ```

5. Compute the anomaly score:

    ```python
    pot_detector.fit()

    anomaly_score_ts = pot_detector.return_dataset("anomaly_score")
    anomaly_score_ts.tail()
    ```
    ```shell
    Date-Time
    2016-10-29 00:00:00    0.0
    2016-10-29 01:00:00    0.0
    2016-10-29 02:00:00    0.0
    2016-10-29 03:00:00    0.0
    2016-10-29 04:00:00    0.0
    Name: Example Dataset, dtype: float64
    ...
    ```

6. The parameters are now stored inside the detector class:

    ```python
    print(pot_detector.params)
    ```
    ```shell
    {0: {'datetime': Timestamp('2016-10-29 03:00:00'),
    'c': 0.0,
    'loc': 0.0,
    'scale': 0.0,
    'p_value': 0.0,
    'anomaly_score': 0.0},
    1: {'datetime': Timestamp('2016-10-29 04:00:00'),
    'c': 0.324456778899,
    'loc': 0,
    'scale': 0.19125308567629334,
    'p_value': 0.19286132173263668,
    'anomaly_score': 5.1850728337654886},
    ...}
    ```

7. Detect the extremely high anomalies:

    ```python
    pot_detector.detect(0.90)

    anomaly_ts = pot_detector.return_dataset("anomaly")
    anomaly_ts.head()
    ```
    ```shell
    Date-Time
    2019-02-09 08:00:00    False
    2019-02-09 09:00:00    False
    2019-02-09 10:00:00    False
    2019-02-09 11:00:00    False
    2019-02-09 12:00:00    False
    Name: Example Dataset, dtype: bool
    ```

8. Evaluate your analysis result with Kolmogorov Smirnov 1 sample test:

    ```python
    pot_detector.evaluate("ks")

    kst_result_df = pot_detector.return_dataset("eval")
    >>> kst_result_df.head()
    ```
    ```shell
        total_nonzero_exceedances	stats_distance	p_value	       c    loc     scale
    0	                     5028	        0.0284	 0.8987 0.003566	  0	 0.140657
    ```

# Reference

* Nakamura, C. (2021, July 13). On Choice of Hyper-parameter in Extreme Value Theory Based on Machine Learning Techniques. arXiv:2107.06074 [cs.LG]. https://doi.org/10.48550/arXiv.2107.06074

* Davis, N., Raina, G., & Jagannathan, K. (2019). LSTM-Based Anomaly Detection: Detection Rules from Extreme Value Theory. In Proceedings of the EPIA Conference on Artificial Intelligence 2019. https://doi.org/10.48550/arXiv.1909.06041

* Arian, H., Poorvasei, H., Sharifi, A., & Zamani, S. (2020, November 13). The Uncertain Shape of Grey Swans: Extreme Value Theory with Uncertain Threshold. arXiv:2011.06693v1 [econ.GN]. https://doi.org/10.48550/arXiv.2011.06693

* Yiannis Kalliantzis. (n.d.). Detect Outliers: Expert Outlier Detection and Insights. Retrieved [23-12-04T15:10:12.000Z], from https://detectoutliers.com/

# Wall of Fame

I am deeply grateful to have met, guided, or even just read some inspirational works from people who motivate me to publish this open-source package as a part of my capstone project at CODE university of applied sciences in Berlin (2023):

* My lovely mother Sarbina Lindenberg
* Adam Roe
* Alessandro Dolci
* Christian Leschinski
* Johanna Kokocinski
* Peter Krauß
