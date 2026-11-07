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


ellipse_axes = load().ellipse_axes


def test_diagonal():
    lengths, D = map(np.asarray, ellipse_axes(np.diag([4.0, 0.25])))
    assert np.allclose(lengths, [2.0, 0.5]), f"반축은 1/sqrt(0.25) = 2, 1/sqrt(4) = 0.5 순서입니다. 현재 {lengths}"
    assert np.isclose(abs(D[1, 0]), 1) and np.isclose(abs(D[0, 1]), 1), "긴 반축은 x2 방향, 짧은 반축은 x1 방향입니다"


@pytest.mark.parametrize("n", [2, 3, 5])
def test_endpoints_on_ellipse(n):
    rng = np.random.default_rng(n)
    B = rng.normal(size=(n, n))
    S = B @ B.T + 0.5 * np.eye(n)                           # 양의 정부호
    lengths, D = map(np.asarray, ellipse_axes(S))
    assert lengths.shape == (n,) and D.shape == (n, n)
    assert np.all(np.diff(lengths) <= 1e-12), f"반축 길이는 내림차순이어야 합니다: {lengths}"
    assert np.allclose(D.T @ D, np.eye(n)), "방향 벡터(열)는 정규직교여야 합니다"
    for i in range(n):
        p = lengths[i] * D[:, i]
        assert np.isclose(p @ S @ p, 1), f"{i}번째 반축 끝점이 타원 위에 있지 않습니다"


def test_rotated():
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    R = np.array([[c, -s], [s, c]])
    S = R @ np.diag([1.0, 9.0]) @ R.T
    lengths, D = map(np.asarray, ellipse_axes(S))
    assert np.allclose(lengths, [1.0, 1.0 / 3.0])
    assert np.isclose(abs(D[:, 0] @ R[:, 0]), 1), "긴 반축은 작은 고윳값 1의 고유벡터 방향입니다"


@pytest.mark.parametrize("S", [
    np.array([[1.0, 2.0], [2.0, 1.0]]),     # 고윳값 -1, 3
    np.diag([1.0, 0.0]),                    # PSD지만 PD 아님
    np.array([[2.0, 1.0], [0.0, 2.0]]),     # 비대칭
])
def test_rejects_invalid(S):
    with pytest.raises(ValueError):
        ellipse_axes(S)
