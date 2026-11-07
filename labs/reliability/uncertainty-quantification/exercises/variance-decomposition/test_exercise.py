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


decompose = load().decompose


def test_hand_example():
    mu = np.array([[1.0], [3.0]])        # 평균 2, 구성원 평균의 분산 1
    var = np.array([[2.0], [4.0]])       # 분산의 평균 3
    a, e, t = decompose(mu, var)
    assert np.allclose(a, [3.0]), f"aleatoric은 구성원 분산의 평균 3이어야 합니다. 현재 {a}"
    assert np.allclose(e, [1.0]), f"epistemic은 구성원 평균의 분산(M으로 나눔) 1이어야 합니다. 현재 {e}"
    assert np.allclose(t, [4.0])


def test_shapes():
    rng = np.random.default_rng(0)
    a, e, t = decompose(rng.normal(size=(5, 7)), rng.uniform(0.1, 1, size=(5, 7)))
    for name, arr in (("aleatoric", a), ("epistemic", e), ("total", t)):
        assert np.shape(arr) == (7,), f"{name}의 shape은 (N,) = (7,)이어야 합니다. 현재 {np.shape(arr)}"


def test_agreeing_members_have_no_epistemic():
    mu = np.tile(np.array([0.5, -1.0, 2.0]), (4, 1))   # 모든 구성원이 같은 평균
    var = np.full((4, 3), 0.3)
    a, e, t = decompose(mu, var)
    assert np.allclose(e, 0.0), "구성원이 모두 같은 평균을 내면 epistemic은 0이어야 합니다"
    assert np.allclose(a, 0.3) and np.allclose(t, 0.3)


def test_matches_mixture_samples():
    rng = np.random.default_rng(1)
    mu = np.array([[0.0, 1.0], [2.0, 1.5], [-1.0, 0.5]])     # (3, 2)
    var = np.array([[0.5, 0.1], [1.0, 0.2], [0.25, 0.3]])     # (3, 2)
    _, _, t = decompose(mu, var)
    comp = rng.integers(0, 3, size=200_000)                    # 혼합분포에서 구성원을 고르게 고른다
    samples = mu[comp] + np.sqrt(var[comp]) * rng.normal(size=(200_000, 2))
    assert np.allclose(t, samples.var(axis=0), rtol=0.02), "total은 혼합분포의 분산과 같아야 합니다"
