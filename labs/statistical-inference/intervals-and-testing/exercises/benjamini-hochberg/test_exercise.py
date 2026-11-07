import importlib.util
import os
from math import erf, sqrt
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


m = load()


def test_hand_example_keeps_input_order():
    p = [0.01, 0.04, 0.03, 0.005, 0.5]
    # 정렬: 0.005 ≤ 0.01, 0.01 ≤ 0.02, 0.03 ≤ 0.03, 0.04 ≤ 0.04, 0.5 > 0.05  →  앞의 4개 기각
    assert list(m.bh_reject(p, q=0.05)) == [True, True, True, True, False]


def test_step_up_rejects_below_largest_passing_rank():
    p = [0.02, 0.03, 0.04, 0.045]
    # 순위 1–3은 조건을 못 넘지만 순위 4가 0.045 ≤ 0.05를 만족하므로 모두 기각한다
    assert m.bh_reject(p, q=0.05).all(), "조건을 만족하는 가장 큰 순위까지 모두 기각합니다(처음 실패에서 멈추지 않음)"


def test_nothing_rejected():
    assert not m.bh_reject([0.2, 0.5, 0.9], q=0.05).any()


def test_returns_boolean_array_of_same_length():
    r = m.bh_reject(np.linspace(0.001, 0.9, 7))
    assert isinstance(r, np.ndarray) and r.dtype == bool and r.shape == (7,)


def test_at_least_as_many_as_bonferroni_and_controls_fdr():
    rng = np.random.default_rng(0)
    fdp = []
    for _ in range(300):
        z = rng.normal(np.r_[np.full(20, 3.0), np.zeros(180)], 1.0)
        p = np.array([1 - erf(abs(v) / sqrt(2)) for v in z])          # 양측 p-value
        r = m.bh_reject(p, q=0.1)
        assert r.sum() >= np.sum(p < 0.1 / len(p))
        fdp.append(r[20:].sum() / max(r.sum(), 1))
    assert np.mean(fdp) < 0.1, "BH 절차의 평균 거짓 발견 비율은 q 이하여야 합니다"
