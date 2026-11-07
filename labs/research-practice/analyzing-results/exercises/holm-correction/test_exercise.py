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


holm = load().holm


def reference(p, alpha):
    """정의를 그대로 따라가는 느린 구현: 작은 p부터 기준과 비교하고 처음 실패하면 멈춘다."""
    p = np.asarray(p, float)
    k, rej = len(p), np.zeros(len(p), bool)
    for j, i in enumerate(np.argsort(p, kind="stable")):
        if p[i] > alpha / (k - j):
            break
        rej[i] = True
    return rej


def test_worked_example():
    rej, adj = holm([0.004, 0.014, 0.020, 0.200])
    assert np.allclose(adj, [0.016, 0.042, 0.042, 0.2]), f"보정 p-value가 다릅니다: {adj}"
    assert list(rej) == [True, True, True, False], "앞의 세 가설을 기각해야 합니다"


def test_keeps_input_order():
    rej, adj = holm([0.200, 0.020, 0.004, 0.014])
    assert np.allclose(adj, [0.2, 0.042, 0.016, 0.042]), "결과를 입력과 같은 순서로 돌려줘야 합니다"
    assert list(rej) == [False, True, True, True]


def test_stops_at_first_failure():
    # 정렬하면 0.01, 0.03, 0.04. 0.03 > 0.05/2에서 멈추므로 0.04는 0.05/1보다 작아도 기각하지 않는다.
    rej, adj = holm([0.01, 0.04, 0.03])
    assert list(rej) == [True, False, False], "처음 기준을 넘는 순간 멈춰야 합니다(step-down)"
    assert np.allclose(adj, [0.03, 0.06, 0.06]), "누적 최댓값으로 앞 순위의 보정값을 이어받아야 합니다"


def test_capped_at_one():
    _, adj = holm([0.6, 0.7, 0.9])
    assert np.all(adj <= 1.0), "보정 p-value는 1을 넘지 않습니다"
    assert np.allclose(adj, [1.0, 1.0, 1.0])


def test_more_powerful_than_bonferroni_and_matches_definition():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = rng.integers(2, 12)
        p = rng.uniform(0, 0.1, size=k) ** rng.uniform(0.5, 2)
        rej, _ = holm(p, alpha=0.05)
        assert np.array_equal(rej, reference(p, 0.05)), "정의(작은 p부터 비교, 실패하면 멈춤)와 결과가 다릅니다"
        assert np.all(rej[p <= 0.05 / k]), "Bonferroni가 기각하는 가설은 Holm도 기각해야 합니다"


def test_alpha_argument():
    rej, _ = holm([0.004, 0.014, 0.020, 0.200], alpha=0.01)
    assert list(rej) == [False, False, False, False], "alpha = 0.01이면 0.004 × 4 = 0.016 > 0.01이므로 아무것도 기각하지 않습니다"
