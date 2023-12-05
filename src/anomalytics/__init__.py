__version__ = "0.1.1"

__all__ = [
    "get_anomaly",
    "get_anomaly_score",
    "get_detector",
    "get_exceedance_peaks_over_threshold",
    "read_ts",
    "set_time_window",
]

from anomalytics.models import get_detector
from anomalytics.stats import get_anomaly, get_anomaly_score, get_exceedance_peaks_over_threshold
from anomalytics.time_series import read_ts
from anomalytics.time_windows import set_time_window
