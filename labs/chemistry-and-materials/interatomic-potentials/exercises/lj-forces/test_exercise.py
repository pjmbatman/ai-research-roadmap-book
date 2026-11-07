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


lj_forces = load().lj_forces


def lj_energy(R):
    d = np.linalg.norm(R[:, None] - R[None], axis=-1)
    iu = np.triu_indices(len(R), 1)
    inv6 = (1 / d[iu]) ** 6
    return np.sum(4 * (inv6 ** 2 - inv6))


def test_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    re = 2 ** (1 / 6)
    R = np.array([[0, 0, 0], [re, 0, 0], [0, re, 0], [0, 0, re], [re, re, 0]], float)
    R = R + rng.normal(size=R.shape) * 0.08
    F = np.asarray(lj_forces(R))
    assert F.shape == R.shape, f"힘의 shape은 {R.shape}여야 합니다. 현재 {F.shape}"
    h = 1e-5
    for i in range(len(R)):
        for k in range(3):
            Rp, Rm = R.copy(), R.copy()
            Rp[i, k] += h
            Rm[i, k] -= h
            num = -(lj_energy(Rp) - lj_energy(Rm)) / (2 * h)
            assert abs(F[i, k] - num) < 1e-5, "해석적 힘이 수치 미분과 다릅니다"


def test_newton_third_law():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(6, 3)) + np.arange(6)[:, None] * 1.2
    assert np.abs(np.asarray(lj_forces(R)).sum(0)).max() < 1e-8, "전체 힘의 합이 0이어야 합니다"


def test_equilibrium_zero_force():
    re = 2 ** (1 / 6)
    R = np.array([[0.0, 0, 0], [re, 0, 0]])
    assert np.abs(np.asarray(lj_forces(R))).max() < 1e-8, "평형 거리 2^(1/6)에서 힘은 0이어야 합니다"


def test_attractive_when_far():
    R = np.array([[0.0, 0, 0], [1.5, 0, 0]])         # 평형보다 멀면 서로 당긴다
    F = np.asarray(lj_forces(R))
    assert F[0, 0] > 0 and F[1, 0] < 0, "평형보다 먼 두 원자는 서로 당겨야 합니다"
