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


best_threshold = load().best_threshold


def total_cost(y, p, t, c_fn, c_fp):
    pred = p >= t
    return c_fn * np.sum(~pred & (y == 1)) + c_fp * np.sum(pred & (y == 0))


def test_hand_example():
    y = np.array([0, 0, 1, 0, 1])
    p = np.array([0.1, 0.3, 0.4, 0.6, 0.9])
    # t=0.4: FN 0, FP 1 (0.6) → 비용 1.  t=0.9: FN 1 → 5.  t=0.1: FP 3 → 3.
    t, cost = best_threshold(y, p, c_fn=5.0, c_fp=1.0)
    assert np.isclose(t, 0.4), f"threshold는 0.4여야 합니다. 현재 {t}"
    assert np.isclose(cost, 1.0), f"총비용은 1이어야 합니다. 현재 {cost}"


def test_predict_none_when_fp_expensive():
    y = np.array([0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    t, cost = best_threshold(y, p, c_fn=1.0, c_fp=100.0)
    assert np.isinf(t), "양성 판정이 모두 손해면 np.inf(아무것도 양성으로 판정하지 않음)를 골라야 합니다"
    assert np.isclose(cost, 1.0)


def test_tie_picks_smallest():
    y = np.array([1, 0, 1])
    p = np.array([0.2, 0.5, 0.8])
    # t=0.2: FP 1 → 1. t=0.5: FN 1, FP 1 → 2. t=0.8: FN 1 → 1. 동점이면 0.2
    t, cost = best_threshold(y, p, c_fn=1.0, c_fp=1.0)
    assert np.isclose(t, 0.2) and np.isclose(cost, 1.0), "비용이 같으면 가장 작은 threshold를 고릅니다"


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = (rng.uniform(size=200) < 0.2).astype(int)
        p = np.round(rng.uniform(size=200) * 0.5 + 0.4 * y, 2)          # 동점 포함
        c_fn, c_fp = rng.uniform(1, 20), rng.uniform(0.5, 2)
        cand = np.append(np.unique(p), np.inf)
        costs = [total_cost(y, p, c, c_fn, c_fp) for c in cand]
        t, cost = best_threshold(y, p, c_fn, c_fp)
        assert np.isclose(cost, min(costs)), "모든 후보 중 최소 비용과 달라요"
        assert np.isclose(t, cand[int(np.argmin(costs))])


def test_calibrated_probabilities_near_theory():
    rng = np.random.default_rng(1)
    n, prior = 5_000, 0.1
    y = (rng.uniform(size=n) < prior).astype(int)
    s = rng.normal(size=n) + 2.0 * y
    lr = np.exp(2.0 * s - 2.0)                                           # N(2,1)과 N(0,1)의 우도비
    p = prior * lr / (prior * lr + 1 - prior)                           # 보정된 확률
    t, _ = best_threshold(y, p, c_fn=9.0, c_fp=1.0)
    assert abs(t - 0.1) < 0.03, f"보정된 확률이면 threshold가 c_fp/(c_fp+c_fn) = 0.1 근처여야 합니다. 현재 {t:.3f}"
