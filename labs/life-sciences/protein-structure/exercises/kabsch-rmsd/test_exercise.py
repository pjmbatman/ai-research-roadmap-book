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


kabsch_rmsd = load().kabsch_rmsd


def rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q          # 행렬식 +1인 회전


def structure(rng, L=30):
    steps = rng.normal(size=(L, 3))
    steps *= 3.8 / np.linalg.norm(steps, axis=1, keepdims=True)
    return np.cumsum(steps, axis=0)                   # 이웃 간격 3.8 Å인 사슬


def test_rigid_motion_gives_zero():
    rng = np.random.default_rng(0)
    X = structure(rng)
    Y = X @ rotation(rng).T + np.array([10.0, -4.0, 2.5])
    assert abs(kabsch_rmsd(Y, X)) < 1e-6, "회전·평행이동만 다른 두 구조의 RMSD는 0이어야 합니다"


def test_mirror_is_not_zero():
    rng = np.random.default_rng(1)
    X = structure(rng)
    M = X * np.array([1.0, 1.0, -1.0])
    assert kabsch_rmsd(M, X) > 1.0, "거울상은 반사 없이 겹쳐지지 않아야 합니다(d = sign det)"


def test_noise_level():
    rng = np.random.default_rng(2)
    X = structure(rng, L=200)
    Y = X @ rotation(rng).T + rng.normal(0, 0.3, X.shape)
    assert 0.4 < kabsch_rmsd(Y, X) < 0.6, "좌표마다 0.3 Å 잡음이면 RMSD는 약 0.3·√3 ≈ 0.52 Å입니다"


def test_not_worse_than_random_rotations():
    rng = np.random.default_rng(3)
    X, Y = structure(rng), structure(rng)
    best = kabsch_rmsd(Y, X)
    Y0, X0 = Y - Y.mean(0), X - X.mean(0)
    for _ in range(200):
        R = rotation(rng)
        assert best <= np.sqrt(np.mean(np.sum((Y0 @ R - X0) ** 2, 1))) + 1e-9, "최적 회전보다 나은 회전이 있으면 안 됩니다"


def test_symmetric_and_float():
    rng = np.random.default_rng(4)
    X, Y = structure(rng), structure(rng)
    r = kabsch_rmsd(X, Y)
    assert isinstance(r, float) or np.ndim(r) == 0, "실수 하나를 반환합니다"
    assert np.isclose(r, kabsch_rmsd(Y, X)), "RMSD는 두 구조의 순서와 무관합니다"
