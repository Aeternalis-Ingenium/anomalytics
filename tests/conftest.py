import numpy as np
import pandas as pd
import pytest


@pytest.fixture(name="get_sample_1_ts", scope="function")
def get_sample_1_ts(request):
    request.cls.sample_1_ts = pd.Series(
        data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range(start="2023-01-01", periods=10)
    )


@pytest.fixture(name="get_sample_1_df", scope="function")
def get_sample_1_df(request):
    request.cls.sample_1_df = pd.DataFrame(
        data={
            "feature_1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "feature_2": [15, 17, 24, 36, 23, 15, 75, 56, 89, 105],
        }
    )


@pytest.fixture(name="get_sample_2_ts", scope="function")
def get_sample_2_ts(request):
    request.cls.sample_2_ts = pd.Series(
        index=pd.date_range(start="2023-01-01", periods=50),
        data=[
            263,
            275,
            56,
            308,
            488,
            211,
            70,
            42,
            67,
            472,
            304,
            297,
            480,
            227,
            453,
            342,
            115,
            115,
            67,
            295,
            9,
            228,
            89,
            225,
            360,
            367,
            418,
            124,
            229,
            12,
            111,
            341,
            209,
            374,
            254,
            322,
            99,
            166,
            435,
            481,
            106,
            438,
            180,
            33,
            30,
            330,
            139,
            17,
            268,
            204000,
        ],
    )


@pytest.fixture(name="get_sample_3_ts", scope="function")
def get_sample_3_ts(request):
    np.random.seed(seed=42)
    request.cls.sample_3_ts = pd.Series(data=np.random.rand(100), index=pd.date_range("2020-01-01", periods=100))


@pytest.fixture(name="get_sample_1_gpd_params", scope="function")
def get_sample_1_gpd_params(request):
    request.cls.sample_1_gpd_params = [{"datetime": "2020-01-01", "c": 0.5, "loc": 0, "scale": 1}]


@pytest.fixture(name="get_sample_1_detection_summary", scope="function")
def get_sample_1_detection_summary(request):
    request.cls.sample_1_detection_summary = pd.DataFrame(
        data={
            "row": [0, 1, 3, 9],
            "datetime": ["2023-01-01", "2023-01-02", "2023-01-04", "2023-01-10"],
            "anomalous_data": [61234, 83421, 69898, 75521],
            "anomaly_score": [7.343, 9.312, 7.884, 8.123],
            "anomaly_threshold": [7.3, 7.3, 7.3, 7.3],
        }
    )
