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


scores = load().scores

LAT = np.arange(-80.0, 81, 20.0)        # (9,)
LON = np.arange(0, 360, 30.0)           # (12,)
RNG = np.random.default_rng(0)
CLIM = RNG.normal(280, 5, (LAT.size, LON.size))
OBS = CLIM + RNG.normal(0, 2, (6, 3, LAT.size, LON.size))     # (N=6, L=3, 9, 12)


def test_perfect_forecast():
    rmse, acc = scores(OBS, OBS, CLIM, LAT)
    assert np.shape(rmse) == (3,) and np.shape(acc) == (3,), "rmse와 acc의 shape은 (L,)이어야 합니다"
    assert np.allclose(rmse, 0) and np.allclose(acc, 1)


def test_damped_anomaly_keeps_acc():
    fc = CLIM + 0.5 * (OBS - CLIM)                             # 편차의 패턴은 같고 크기만 절반
    rmse, acc = scores(fc, OBS, CLIM, LAT)
    rmse_clim, _ = scores(OBS * 0 + CLIM + 1e-9, OBS, CLIM, LAT)
    assert np.allclose(acc, 1), "ACC는 편차의 크기가 아니라 패턴의 상관이다"
    assert np.allclose(rmse, 0.5 * rmse_clim, rtol=1e-6)


def test_latitude_weighting():
    fc = OBS.copy()
    fc[..., 0, :] += 3.0                                       # 80°S 줄에만 오차
    r_pole, _ = scores(fc, OBS, CLIM, LAT)
    fc = OBS.copy()
    fc[..., 4, :] += 3.0                                       # 적도 줄에만 같은 오차
    r_eq, _ = scores(fc, OBS, CLIM, LAT)
    w = np.cos(np.deg2rad(LAT))
    w = w / w.mean()
    assert np.allclose(r_pole, np.sqrt(w[0] * 9 / LAT.size)), "고위도 오차는 cos(위도)만큼 작게 반영되어야 합니다"
    assert np.allclose(r_eq, np.sqrt(w[4] * 9 / LAT.size))


def test_matches_loop():
    fc = CLIM + RNG.normal(0, 2, OBS.shape) + 0.6 * (OBS - CLIM)
    rmse, acc = scores(fc, OBS, CLIM, LAT)
    w = np.cos(np.deg2rad(LAT))[:, None] * np.ones(LON.size)
    w = w / w.mean()
    for l in range(3):
        mses, accs = [], []
        for n in range(6):
            e = fc[n, l] - OBS[n, l]
            mses.append((w * e**2).mean())
            a, b = fc[n, l] - CLIM, OBS[n, l] - CLIM
            accs.append((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))
        assert np.isclose(rmse[l], np.sqrt(np.mean(mses))), "예보마다 MSE를 구해 평균한 뒤 제곱근을 취합니다"
        assert np.isclose(acc[l], np.mean(accs))
