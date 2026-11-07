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


weighted_mean = load().weighted_mean

LAT = np.arange(-89.5, 90, 1.0)     # (180,) 1도 격자
LON = np.arange(0, 360, 2.0)        # (180,)


def test_constant_field():
    f = np.full((LAT.size, LON.size), 7.0)
    assert np.isclose(weighted_mean(f, LAT), 7.0), "상수장의 평균은 그 상수여야 합니다"


def test_matches_area_weighting():
    f = np.repeat((300 - 45 * np.sin(np.deg2rad(LAT)) ** 2)[:, None], LON.size, axis=1)
    m = float(weighted_mean(f, LAT))
    assert abs(m - 285.0) < 0.01, f"300 - 45 sin^2(위도)의 면적 평균은 285에 가까워야 합니다. 현재 {m:.3f}"
    assert abs(m - f.mean()) > 5, "단순 평균과 같습니다. cos(위도) 가중치를 썼는지 확인하세요"


def test_nan_excluded_from_both_sums():
    f = np.full((LAT.size, LON.size), 10.0)
    f[:, :30] = np.nan
    m = float(weighted_mean(f, LAT))
    assert np.isclose(m, 10.0), f"결측을 0으로 다루면 평균이 10보다 작아집니다. 현재 {m:.3f}"


def test_leading_axes_kept():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(4, 3, LAT.size, LON.size))
    f[0, 0, :10, :] = np.nan
    out = np.asarray(weighted_mean(f, LAT))
    assert out.shape == (4, 3), f"앞쪽 축은 남아야 합니다. 현재 shape {out.shape}"
    w = np.cos(np.deg2rad(LAT))[:, None] * np.ones(LON.size)
    ok = ~np.isnan(f[0, 0])
    assert np.isclose(out[0, 0], (w * f[0, 0])[ok].sum() / w[ok].sum())
    assert np.isclose(out[2, 1], (w * f[2, 1]).sum() / w.sum())


def test_all_missing_is_nan():
    f = np.full((LAT.size, LON.size), np.nan)
    assert np.isnan(weighted_mean(f, LAT)), "모든 칸이 결측이면 NaN이어야 합니다"
