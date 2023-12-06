from unittest import TestCase

from anomalytics import set_time_window
from anomalytics.time_windows import compute_pot_windows


class TestTimeWindow(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ht0, self.ht1, self.ht2 = set_time_window(
            total_rows=1000, method="POT", analysis_type="historical", t0_pct=0.65, t1_pct=0.25, t2_pct=0.10
        )
        self.rtt0, self.rtt1, self.rtt2 = set_time_window(
            total_rows=1000, method="POT", analysis_type="real-time", t0_pct=0.7, t1_pct=0.3, t2_pct=0.0
        )

    def test_compute_pot_windows_executed_correctly_via_set_time_window(self):
        expected_t0, expected_t1, expected_t2 = compute_pot_windows(
            total_rows=1000, analysis_type="historical", t0_pct=0.65, t1_pct=0.25, t2_pct=0.10
        )
        self.assertEqual(first=(self.ht0 + self.ht1 + self.ht2), second=(expected_t0 + expected_t1 + expected_t2))
        self.assertEqual(first=self.ht0, second=expected_t0)
        self.assertEqual(first=self.ht1, second=expected_t1)
        self.assertEqual(first=self.ht2, second=expected_t2)

    def test_pot_windows_for_historical_analysis_via_set_time_window(self):
        self.assertEqual(first=(self.ht0 + self.ht1 + self.ht2), second=1000)
        self.assertEqual(first=self.ht0, second=650)
        self.assertEqual(first=self.ht1, second=250)
        self.assertEqual(first=self.ht2, second=100)

    def test_pot_windows_for_realtime_analysis_via_set_time_window(self):
        self.assertEqual(first=(self.rtt0 + self.rtt1 + self.rtt2), second=1000)
        self.assertEqual(first=self.rtt0, second=699)
        self.assertEqual(first=self.rtt1, second=300)
        self.assertEqual(first=self.rtt2, second=1)

    def tearDown(self) -> None:
        return super().tearDown()
