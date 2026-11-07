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


trend_test = load().trend_test


def ar1(n, phi, seed):
    g = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = g.normal()
    for i in range(1, n):
        y[i] = phi * y[i - 1] + np.sqrt(1 - phi**2) * g.normal()
    return y


def test_slope_matches_polyfit():
    y = 0.02 * np.arange(60) + ar1(60, 0.5, 0)
    b, se, n_eff, se_adj = trend_test(y)
    assert np.isclose(b, np.polyfit(np.arange(60), y, 1)[0]), "기울기는 최소제곱 기울기와 같아야 합니다"
    assert se > 0 and se_adj >= se


def test_white_noise_keeps_n():
    y = np.random.default_rng(1).normal(size=3000)
    b, se, n_eff, se_adj = trend_test(y)
    assert n_eff > 0.9 * 3000, f"독립 잡음이면 유효 표본 수는 n에 가까워야 합니다. 현재 {n_eff:.0f}"
    assert np.isclose(se_adj, se, rtol=0.06)


def test_ar1_shrinks_n():
    y = ar1(4000, 0.6, 2)
    b, se, n_eff, se_adj = trend_test(y)
    assert abs(n_eff / 4000 - 0.4 / 1.6) < 0.03, f"phi = 0.6이면 n_eff/n은 약 0.25입니다. 현재 {n_eff / 4000:.3f}"
    assert np.isclose(se_adj, se * np.sqrt(3998 / (n_eff - 2)))


def test_negative_r1_is_clipped():
    y = 0.01 * np.arange(40) + np.where(np.arange(40) % 2 == 0, 1.0, -1.0)
    _, _, n_eff, _ = trend_test(y)
    assert n_eff == 40, "r1 < 0이어도 n_eff는 n을 넘지 않아야 합니다"


def test_manual_values():
    y = np.array([0.1, 0.4, 0.2, 0.6, 0.9, 0.7, 1.1, 1.0])
    b, se, n_eff, se_adj = trend_test(y)
    t = np.arange(8) - 3.5
    e = y - y.mean() - b * t
    r1 = (e[1:] * e[:-1]).sum() / (e**2).sum()
    assert np.isclose(b, (t * y).sum() / (t**2).sum())
    assert np.isclose(se, np.sqrt((e**2).sum() / 6 / (t**2).sum()))
    assert np.isclose(n_eff, np.clip(8 * (1 - r1) / (1 + r1), 3, 8))
