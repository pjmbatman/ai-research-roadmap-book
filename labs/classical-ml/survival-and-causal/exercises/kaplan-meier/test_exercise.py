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


mod = load()


def test_hand_example():
    time = np.array([3, 5, 5, 8, 10, 12])
    event = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
    t_j, S = mod.kaplan_meier(time, event)
    # t=3: 6명 중 1 → 5/6. t=5: 5명(중도절단 포함) 중 1 → ×4/5. t=8: 3명 중 1 → ×2/3. t=12: 1명 중 1 → 0
    expect = np.cumprod([5 / 6, 4 / 5, 2 / 3, 0.0])
    assert np.allclose(t_j, [3, 5, 8, 12]), f"고장이 관측된 시각만 써야 합니다. 현재 {np.asarray(t_j).tolist()}"
    assert np.allclose(S, expect), f"S는 {np.round(expect, 4).tolist()}이어야 합니다. 현재 {np.round(S, 4).tolist()}"


def test_tied_censoring_in_risk_set():
    time = np.array([2, 2, 2, 4])
    event = np.array([1, 0, 0, 1], dtype=bool)
    _, S = mod.kaplan_meier(time, event)
    assert np.isclose(S[0], 3 / 4), "같은 시각에 중도절단된 셀도 그 시각의 위험 집합(4개)에 들어가야 합니다"


def test_no_censoring_equals_empirical():
    rng = np.random.default_rng(0)
    time = rng.exponential(10, 50)
    t_j, S = mod.kaplan_meier(time, np.ones(50, dtype=bool))
    emp = np.array([np.mean(time > t) for t in t_j])
    assert np.allclose(S, emp), "중도절단이 없으면 KM은 경험 생존함수 P(T > t)와 같아야 합니다"


def test_median():
    t_j, S = np.array([1.0, 2.0, 3.0]), np.array([0.8, 0.5, 0.2])
    assert mod.km_median(t_j, S) == 2.0, "S가 처음 0.5 이하가 되는 시각을 반환합니다"
    assert np.isinf(mod.km_median(t_j, np.array([0.9, 0.8, 0.7]))), "0.5 아래로 내려가지 않으면 np.inf입니다"


def test_heavy_censoring_simulation():
    rng = np.random.default_rng(1)
    life = 1000 * rng.weibull(3.0, 2000)
    stop = rng.uniform(300, 1500, 2000)
    time, event = np.minimum(life, stop), life <= stop
    t_j, S = mod.kaplan_meier(time, event)
    for t in (500, 800, 1000):
        k = np.searchsorted(t_j, t, side="right")
        assert abs(S[k - 1] - np.exp(-(t / 1000) ** 3)) < 0.05, f"t={t}에서 참 생존함수와 너무 멉니다"
    true_med = 1000 * np.log(2) ** (1 / 3)
    assert abs(mod.km_median(t_j, S) - true_med) < 50
