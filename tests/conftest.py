import numpy as np
import pandas as pd
import pytest


@pytest.fixture(name="get_sample_1_ts", scope="function")
def get_sample_1_ts(request):
    request.cls.sample_1_ts = pd.Series(
        data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range(start="2023-01-01", periods=10)
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
