import unittest

import numpy as np
import pandas as pd

from anomalytics.evals import ks_1sample


class TestEvaluationMethod(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sample_ts = pd.Series(np.random.rand(100), index=pd.date_range("2020-01-01", periods=100))
        self.fit_params = [{"datetime": self.sample_ts.index[-1], "c": 0.5, "loc": 0, "scale": 1}]

    def test_ks_1sample_with_valid_input_pot(self):
        result = ks_1sample(ts=self.sample_ts, stats_method="POT", fit_params=self.fit_params)
        self.assertIsInstance(result, dict)

        self.assertIn("total_nonzero_exceedances", result)
        self.assertIn("start_datetime", result)
        self.assertIn("end_datetime", result)
        self.assertIn("stats_distance", result)
        self.assertIn("p_value", result)

    def test_ks_1sample_with_invalid_series(self):
        with self.assertRaises(TypeError):
            ks_1sample(ts="not a series", stats_method="POT", fit_params=self.fit_params)

    def test_ks_1sample_with_unimplemented_method(self):
        with self.assertRaises(NotImplementedError):
            ks_1sample(ts=self.sample_ts, stats_method="AE", fit_params={})

    def tearDown(self) -> None:
        return super().tearDown()
