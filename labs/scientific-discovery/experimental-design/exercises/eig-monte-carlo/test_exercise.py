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


eig_nmc = load().eig_nmc


def line_design(x):                                       # y = theta0 + theta1 x
    return np.stack([np.ones_like(x), x], axis=1)        # (n, 2)


def eig_exact(Phi, sigma):                                # 사전 N(0, I)
    return 0.5 * np.linalg.slogdet(np.eye(len(Phi)) + Phi @ Phi.T / sigma**2)[1]


@pytest.mark.parametrize("x", [[0.0, 0.0], [-1.0, 1.0], [-1.0, 0.0, 1.0]])
def test_matches_linear_gaussian(x):
    Phi = line_design(np.array(x))
    rng = np.random.default_rng(0)
    est = eig_nmc(lambda th: th @ Phi.T, rng.normal(size=(1500, 2)), rng.normal(size=(1500, 2)), 0.5, rng)
    exact = eig_exact(Phi, 0.5)
    assert abs(est - exact) < 0.1, f"해석해 {exact:.3f}와 달라도 너무 다릅니다: {est:.3f}"


def test_uninformative_design_is_near_zero():
    rng = np.random.default_rng(1)
    zero = lambda th: np.zeros((len(th), 3))              # 측정 평균이 θ와 무관
    est = eig_nmc(zero, rng.normal(size=(800, 1)), rng.normal(size=(800, 1)), 1.0, rng)
    assert abs(est) < 0.02, f"측정이 θ와 무관하면 EIG는 0이어야 합니다. 현재 {est:.3f}"


def test_more_repeats_more_information():
    rng = np.random.default_rng(2)
    ests = []
    for n in (1, 4):
        Phi = np.ones((n, 1))                             # y_i = θ + ε, n번 반복
        ests.append(eig_nmc(lambda th: th @ Phi.T, rng.normal(size=(1500, 1)), rng.normal(size=(1500, 1)), 1.0, rng))
    assert ests[1] > ests[0] + 0.3, f"반복 측정이 늘면 EIG가 커져야 합니다. 현재 {ests}"
    assert abs(ests[0] - 0.5 * np.log(2)) < 0.08, "1번 측정의 EIG는 0.5 log 2에 가까워야 합니다"


def test_returns_scalar():
    rng = np.random.default_rng(3)
    est = eig_nmc(lambda th: th, rng.normal(size=(50, 1)), rng.normal(size=(60, 1)), 1.0, rng)
    assert np.ndim(est) == 0, "실수 하나를 반환해야 합니다"
