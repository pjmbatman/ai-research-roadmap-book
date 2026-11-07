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


best_split = load().best_split


def sse_at(x, y, t):
    left, right = y[x <= t], y[x > t]
    return ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()


def brute_force(x, y):
    u = np.unique(x)
    cands = (u[:-1] + u[1:]) / 2
    sses = np.array([sse_at(x, y, t) for t in cands])
    i = np.argmin(sses)
    return cands[i], sses[i]


def test_hand_example():
    t, sse = best_split([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 5.0, 5.0])
    assert np.isclose(t, 2.5), f"문턱은 2와 3의 중점 2.5여야 합니다. 현재 {t}"
    assert np.isclose(sse, 0.0), "두 쪽이 각각 상수이면 SSE는 0입니다"


def test_ties_stay_together():
    x = np.array([1.0, 2.0, 2.0, 5.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    t, sse = best_split(x, y)
    assert t in (1.5, 3.5), f"후보 문턱은 서로 다른 값의 중점(1.5, 3.5)뿐입니다. 현재 {t}"
    assert np.isclose(sse, sse_at(x, y, t))


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = np.round(rng.uniform(0, 10, 60), 1)                    # 같은 값이 섞인 특징
        y = np.where(x > 6.3, 3.0, 0.0) + rng.normal(0, 0.5, 60)
        t, sse = best_split(x, y)
        t_ref, sse_ref = brute_force(x, y)
        assert np.isclose(sse, sse_ref), "SSE가 전수 탐색의 최솟값과 다릅니다"
        assert np.isclose(t, t_ref), "문턱이 전수 탐색 결과와 다릅니다"


def test_monotone_transform_gives_same_partition():
    rng = np.random.default_rng(1)
    x = rng.uniform(1, 100, 80)
    y = np.sin(x / 15) + rng.normal(0, 0.1, 80)
    t1, s1 = best_split(x, y)
    t2, s2 = best_split(np.log(x), y)
    assert np.array_equal(x <= t1, np.log(x) <= t2), "단조 증가 변환을 해도 같은 샘플 분할이 나와야 합니다"
    assert np.isclose(s1, s2)
