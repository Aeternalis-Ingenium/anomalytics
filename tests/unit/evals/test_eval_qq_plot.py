import unittest
from unittest.mock import patch

import pytest

from anomalytics.evals import calculate_theoretical_q, visualize_qq_plot


@pytest.mark.usefixtures("get_sample_3_ts", "get_sample_1_gpd_params")
class TestQQPlotEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_calculate_theoretical_q_pot(self):
        sorted_nonzero_ts, theoretical_q, params = calculate_theoretical_q(
            ts=self.sample_3_ts, stats_method="POT", fit_params=self.sample_1_gpd_params  # type: ignore
        )

        self.assertIsInstance((sorted_nonzero_ts, theoretical_q, params), tuple)
        self.assertEqual(len(sorted_nonzero_ts), len(theoretical_q))
        self.assertEqual(params, self.sample_1_gpd_params[0])  # type: ignore

    def test_calculate_theoretical_q_with_invalid_series(self):
        with self.assertRaises(TypeError):
            calculate_theoretical_q(ts="not a series", stats_method="POT", fit_params=self.sample_1_gpd_params)  # type: ignore

    def test_calculate_theoretical_q_with_unimplemented_method(self):
        with self.assertRaises(NotImplementedError):
            calculate_theoretical_q(ts=self.sample_3_ts, stats_method="AE", fit_params=[{}])  # type: ignore

    @patch("matplotlib.pyplot.show")
    def test_visualize_qq_plot_with_valid_input(self, mock_show):
        visualize_qq_plot(ts=self.sample_3_ts, stats_method="POT", fit_params=self.sample_1_gpd_params)  # type: ignore
        mock_show.assert_called_once()

    def test_visualize_qq_plot_with_invalid_series(self):
        with self.assertRaises(TypeError):
            visualize_qq_plot(ts="not a series", stats_method="POT", fit_params=self.sample_1_gpd_params)  # type: ignore

    def test_visualize_qq_plot_with_unimplemented_method(self):
        with self.assertRaises(NotImplementedError):
            visualize_qq_plot(ts=self.sample_3_ts, stats_method="AE", fit_params=self.sample_1_gpd_params)  # type: ignore

    def tearDown(self) -> None:
        return super().tearDown()
