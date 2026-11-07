import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


monthly_anomaly = load().monthly_anomaly

YEARS = np.arange(1981, 2021)
MONTH = np.tile(np.arange(12), YEARS.size)          # (480,)
YEAR = np.repeat(YEARS, 12)                          # (480,)
BASE = (YEAR >= 1991) & (YEAR <= 2020)


def test_pure_seasonal_cycle_gives_zero_anomaly():
    v = 10 * np.sin(2 * np.pi * MONTH / 12)
    anom, clim = monthly_anomaly(v, MONTH, BASE)
    assert np.shape(clim) == (12,), f"clim의 shape은 (12,)여야 합니다. 현재 {np.shape(clim)}"
    assert np.allclose(anom, 0), "계절 주기만 있는 자료의 편차는 0이어야 합니다"


def test_base_period_only():
    v = 0.1 * (YEAR - 1981.0)                        # 추세만 있는 자료
    anom, clim = monthly_anomaly(v, MONTH, BASE)
    assert np.allclose(clim, 0.1 * (2005.5 - 1981)), "기후값은 기준 기간(1991–2020)의 평균이어야 합니다"
    assert np.isclose(anom[BASE].mean(), 0), "기준 기간 편차의 평균은 0이어야 합니다"
    assert anom[~BASE].mean() < -1.5, "기준 기간 밖의 시점에서도 그 달의 기후값을 빼야 합니다"


def test_nan_in_base_ignored():
    rng = np.random.default_rng(1)
    v = rng.normal(size=MONTH.size)
    v[5] = np.nan
    v[np.flatnonzero(BASE)[0]] = np.nan             # 기준 기간의 1월 하나가 결측
    anom, clim = monthly_anomaly(v, MONTH, BASE)
    assert np.all(np.isfinite(clim)), "기준 기간의 결측 때문에 기후값이 NaN이 되면 안 됩니다"
    assert np.isnan(anom[5]) and np.isfinite(anom[6])


def test_gridded_values():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(MONTH.size, 3, 4)) + np.arange(12)[MONTH][:, None, None]
    anom, clim = monthly_anomaly(v, MONTH, BASE)
    assert np.shape(clim) == (12, 3, 4) and np.shape(anom) == v.shape
    assert np.allclose(clim[7], v[BASE & (MONTH == 7)].mean(axis=0))
    assert np.allclose(anom, v - clim[MONTH])
