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


d_optimal = load().d_optimal
CAND = np.linspace(-1, 1, 21)


def line(x):
    x = np.asarray(x, dtype=float)
    return np.stack([np.ones_like(x), x], axis=1)


def quad(x):
    x = np.asarray(x, dtype=float)
    return np.stack([np.ones_like(x), x, x**2], axis=1)


def logdet(features, x):
    return np.linalg.slogdet(features(x).T @ features(x))[1]


def test_line_uses_endpoints():
    x = d_optimal(line, CAND, 4, np.random.default_rng(0))
    assert np.allclose(x, [-1, -1, 1, 1]), f"직선의 D-최적 설계는 양 끝에 두 번씩입니다. 현재 {x}"


def test_quadratic_three_support_points():
    for seed in range(3):
        x = d_optimal(quad, CAND, 6, np.random.default_rng(seed))
        assert np.allclose(x, [-1, -1, 0, 0, 1, 1]), f"2차 모델의 D-최적 설계는 -1, 0, 1에 두 번씩입니다. 현재 {x}"


def test_beats_uniform_design():
    x = d_optimal(quad, CAND, 5, np.random.default_rng(4))
    assert np.shape(x) == (5,), "실험 수만큼의 조건을 반환해야 합니다"
    assert np.all(np.diff(x) >= 0), "설계를 정렬해 반환해야 합니다"
    assert logdet(quad, x) >= logdet(quad, np.linspace(-1, 1, 5)) - 1e-9, "등간격 설계보다 log det가 작습니다"


def test_handles_singular_start():
    rng = np.random.default_rng(2)                       # 초기 설계 [1, 0, 0]: 2차 모델에서 특이 행렬
    x = d_optimal(quad, np.array([0.0, 0.0, 0.5, 1.0]), 3, rng)
    assert np.allclose(x, [0.0, 0.5, 1.0]), f"특이 행렬에서 시작해도 서로 다른 세 점으로 빠져나와야 합니다. 현재 {x}"
