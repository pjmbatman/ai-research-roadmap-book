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


m = load()


def rotated(eigs, seed):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(len(eigs), len(eigs))))
    return Q @ np.diag(eigs) @ Q.T


def test_gd_one_step():
    H, b = np.diag([1.0, 4.0]), np.array([1.0, 0.0])
    x = m.gd_quadratic(H, b, np.zeros(2), 0.1, 1)
    assert np.allclose(x, [0.1, 0.0]), "x1 = x0 - lr * (H x0 - b) 입니다"


def test_diagonal_value():
    assert np.isclose(m.max_stable_lr(np.diag([1.0, 10.0])), 0.2), "고윳값이 1, 10이면 상한은 2/10 = 0.2입니다"


@pytest.mark.parametrize("eigs,seed", [([1.0, 10.0], 0), ([0.5, 3.0, 40.0], 1), ([2.0, 2.0, 5.0, 100.0], 2)])
def test_boundary(eigs, seed):
    H = rotated(eigs, seed)
    b = np.random.default_rng(seed).normal(size=len(eigs))
    x_star = np.linalg.solve(H, b)
    x0 = x_star + np.random.default_rng(seed + 10).normal(size=len(eigs))
    lr_max = m.max_stable_lr(H)
    below = m.gd_quadratic(H, b, x0, 0.99 * lr_max, 3000)
    above = m.gd_quadratic(H, b, x0, 1.01 * lr_max, 3000)
    assert np.linalg.norm(below - x_star) < 1e-3, "상한보다 조금 작은 학습률에서는 수렴해야 합니다"
    assert np.linalg.norm(above - x_star) > 1e3, "상한보다 조금 큰 학습률에서는 발산해야 합니다"


def test_depends_on_largest_curvature_only():
    a = m.max_stable_lr(np.diag([0.001, 10.0]))
    c = m.max_stable_lr(np.diag([5.0, 10.0]))
    assert np.isclose(a, c), "상한은 가장 큰 고윳값(최대 곡률)만으로 정해집니다"
