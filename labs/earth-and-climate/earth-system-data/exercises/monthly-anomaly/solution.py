import numpy as np


def monthly_anomaly(values, month, base):
    """기준 기간의 월별 기후값과 전체 기간의 편차.

    values: (T, ...), month: (T,) 0–11, base: (T,) bool
    반환: anom (T, ...), clim (12, ...)
    """
    values = np.asarray(values, dtype=float)
    month = np.asarray(month)
    base = np.asarray(base, dtype=bool)
    clim = np.stack([np.nanmean(values[base & (month == m)], axis=0) for m in range(12)])  # (12, ...)
    return values - clim[month], clim
