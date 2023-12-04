__version__ = "0.1.0"

__all__ = ["set_time_window"]

from anomalytics.stats import get_exceedance_peaks_over_threshold
from anomalytics.time_series import read_ts
from anomalytics.time_windows import set_time_window
