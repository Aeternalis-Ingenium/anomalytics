import unittest

import pytest

from anomalytics.evals import ks_1sample


@pytest.mark.usefixtures("get_sample_3_ts", "get_sample_1_gpd_params")
class TestKolmogorovSmirnovEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_ks_1sample_with_valid_input_pot(self):
        result = ks_1sample(dataset=self.sample_3_ts, stats_method="POT", fit_params=self.sample_1_gpd_params)  # type: ignore
        self.assertIsInstance(result, dict)

        self.assertIn("total_nonzero_exceedances", result)
        self.assertIn("stats_distance", result)
        self.assertIn("p_value", result)

    def test_ks_1sample_with_invalid_dataset(self):
        with self.assertRaises(TypeError):
            ks_1sample(dataset="not a series", stats_method="POT", fit_params=self.sample_1_gpd_params)  # type: ignore

    def test_ks_1sample_with_invalid_fit_params(self):
        with self.assertRaises(TypeError):
            ks_1sample(dataset=self.sample_3_ts, stats_method="AE", fit_params={})  # type: ignore

    def tearDown(self) -> None:
        return super().tearDown()
