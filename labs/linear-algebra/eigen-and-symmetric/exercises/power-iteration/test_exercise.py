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


power_iteration = load().power_iteration


def same_direction(u, v):
    return abs(abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)) - 1) < 1e-6


@pytest.mark.parametrize("n", [3, 6, 10])
def test_symmetric(n):
    rng = np.random.default_rng(n)
    Qm, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.linspace(1.0, 5.0, n)                          # 지배 고윳값 5, 둘째 4.x
    S = Qm @ np.diag(eigs) @ Qm.T
    lam, v = power_iteration(S)
    v = np.asarray(v)
    assert v.shape == (n,), f"v의 shape은 ({n},)이어야 합니다. 현재 {v.shape}"
    assert np.isclose(np.linalg.norm(v), 1), "v는 단위 벡터여야 합니다"
    assert np.isclose(lam, 5.0, atol=1e-6), f"지배 고윳값은 5입니다. 현재 {lam}"
    assert same_direction(v, Qm[:, -1]), "고유벡터 방향이 다릅니다(부호는 무시합니다)"


def test_eigen_equation():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(5, 5))
    S = B @ B.T                                              # PSD
    lam, v = power_iteration(S)
    assert np.allclose(S @ v, lam * v, atol=1e-5), "A v = lam v를 만족해야 합니다"


def test_leslie():
    L = np.array([[0.0, 1.5, 1.2], [0.6, 0.0, 0.0], [0.0, 0.4, 0.0]])
    lam, v = power_iteration(L)
    assert np.isclose(lam, 1.0801, atol=1e-4), f"Leslie 행렬의 지배 고윳값은 약 1.0801입니다. 현재 {lam}"
    dist = np.abs(v) / np.abs(v).sum()
    assert np.allclose(dist, [0.5678, 0.3154, 0.1168], atol=1e-3), f"안정 연령 분포가 다릅니다: {dist}"


def test_negative_dominant():
    rng = np.random.default_rng(5)
    Qm, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    A = Qm @ np.diag([-3.0, 1.0]) @ Qm.T
    lam, v = power_iteration(A)
    assert np.isclose(lam, -3.0, atol=1e-6), f"지배 고윳값은 -3입니다(절댓값이 가장 큼). 현재 {lam}"
    assert same_direction(v, Qm[:, 0])
