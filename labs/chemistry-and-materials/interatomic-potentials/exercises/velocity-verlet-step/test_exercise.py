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


verlet_step = load().verlet_step


def lj_energy(R):
    d = np.linalg.norm(R[:, None] - R[None], axis=-1)
    iu = np.triu_indices(len(R), 1)
    inv6 = (1 / d[iu]) ** 6
    return np.sum(4 * (inv6 ** 2 - inv6))


def lj_forces(R):
    diff = R[:, None] - R[None]
    d = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d, np.inf)
    inv6 = (1 / d) ** 6
    coeff = 24 * (2 * inv6 ** 2 - inv6) / d ** 2
    return np.einsum("ij,ijk->ik", coeff, diff)


def test_harmonic_one_step():
    k = 2.0
    force_fn = lambda R: -k * R
    R = np.array([[1.0, 0.0, 0.0]])
    V = np.array([[0.0, 0.5, 0.0]])
    dt = 0.1
    Vh = V + 0.5 * dt * force_fn(R)
    R_exp = R + dt * Vh
    V_exp = Vh + 0.5 * dt * force_fn(R_exp)
    R_new, V_new = verlet_step(R, V, force_fn, dt)
    assert np.allclose(R_new, R_exp), "위치 갱신이 식과 다릅니다"
    assert np.allclose(V_new, V_exp), "속도 갱신이 식과 다릅니다. 힘을 두 번(현재·갱신된 위치) 쓰는지 확인하세요"


def test_zero_force_is_uniform_motion():
    force_fn = lambda R: np.zeros_like(R)
    R = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    V = np.array([[0.3, -0.2, 0.1], [0.0, 0.4, 0.0]])
    R_new, V_new = verlet_step(R, V, force_fn, 0.5)
    assert np.allclose(R_new, R + 0.5 * V), "힘이 0이면 등속 운동이어야 합니다"
    assert np.allclose(V_new, V), "힘이 0이면 속도가 그대로여야 합니다"


def test_energy_conservation():
    re = 2 ** (1 / 6)
    R = np.array([[0.0, 0, 0], [1.2, 0, 0], [0.6, 1.0, 0]])
    V = np.array([[0.05, 0, 0], [-0.05, 0.02, 0], [0.0, -0.02, 0]])
    E0 = 0.5 * np.sum(V ** 2) + lj_energy(R)
    for _ in range(2000):
        R, V = verlet_step(R, V, lj_forces, 0.002)
    E = 0.5 * np.sum(V ** 2) + lj_energy(R)
    assert abs(E - E0) < 1e-2, f"총에너지가 보존되지 않습니다: {E0:.4f} -> {E:.4f}"
