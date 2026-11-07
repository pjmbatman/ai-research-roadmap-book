import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()
sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
s = rng.normal(size=60000)
p_cal = sig(-3.0 + 1.5 * s)                     # 보정된 확률
y = (rng.uniform(size=s.size) < p_cal).astype(int)
grid = np.linspace(0.005, 0.6, 120)


def test_bayes_threshold():
    assert m.bayes_threshold(1.0, 9.0) == pytest.approx(0.1)
    assert m.bayes_threshold(1.0, 1.0) == pytest.approx(0.5), "비용이 같으면 0.5입니다"


def test_mean_cost_hand():
    p = np.array([0.9, 0.2, 0.6, 0.05])
    yy = np.array([0, 1, 1, 0])
    # thr 0.5: 경보 = [T, F, T, F] → 오경보 1개(비용 1), 미탐지 1개(비용 5) → 평균 6/4
    assert m.mean_cost(p, yy, 0.5, 1.0, 5.0) == pytest.approx(1.5)


def test_best_threshold_matches_theory_when_calibrated():
    c_fp, c_fn = 1.0, 19.0
    thr = m.best_threshold(p_cal, y, c_fp, c_fn, grid)
    assert abs(thr - m.bayes_threshold(c_fp, c_fn)) < 0.03, f"보정된 확률이면 이론값 0.05 근처여야 합니다. 현재 {thr}"
    assert m.mean_cost(p_cal, y, 0.05, c_fp, c_fn) < m.mean_cost(p_cal, y, 0.5, c_fp, c_fn)


def test_empirical_threshold_helps_when_miscalibrated():
    p_bad = sig(2.5 * (-3.0 + 1.5 * s) + 4.0)    # 순서는 같지만 보정이 틀린 확률
    c_fp, c_fn = 1.0, 19.0
    thr = m.best_threshold(p_bad, y, c_fp, c_fn, grid)
    assert m.mean_cost(p_bad, y, thr, c_fp, c_fn) < m.mean_cost(p_bad, y, m.bayes_threshold(c_fp, c_fn), c_fp, c_fn)
