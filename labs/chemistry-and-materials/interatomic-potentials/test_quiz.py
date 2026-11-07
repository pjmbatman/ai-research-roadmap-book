"""28.3 퀴즈(content/28-chemistry-and-materials/03-interatomic-potentials/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/28-chemistry-and-materials/03-interatomic-potentials/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


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


def verlet(R, V, force_fn, dt, steps):
    F = force_fn(R)
    Es = []
    for _ in range(steps):
        V = V + 0.5 * dt * F
        R = R + dt * V
        F = force_fn(R)
        V = V + 0.5 * dt * F
        Es.append(0.5 * np.sum(V ** 2) + lj_energy(R))
    return np.array(Es)


def test_forces_from_energy():
    assert correct("forces-from-energy") == [0]


def test_forces_more_informative():
    assert correct("forces-more-informative") == [0]
    assert 30 * 3 + 1 == 91                                 # 힘 성분 + 에너지


def test_non_conservative():
    assert correct("non-conservative") == [0]
    # 비보존력(회전형)은 에너지를 늘리고, 보존력은 보존한다
    re = 2 ** (1 / 6)
    cluster = np.array([[0, 0, 0], [re, 0, 0], [-re, 0, 0], [0, re, 0], [0, -re, 0]], float)
    R0 = cluster + np.random.default_rng(1).normal(size=cluster.shape) * 0.03
    E_cons = verlet(R0.copy(), np.zeros_like(R0), lj_forces, 0.004, 3000)
    omega = np.array([0, 0, 1.0])
    ncf = lambda R: lj_forces(R) + 0.15 * np.cross(omega, R - R.mean(0))
    E_ncons = verlet(R0.copy(), np.zeros_like(R0), ncf, 0.004, 3000)
    assert abs(E_cons[-1] - E_cons[0]) < 0.05               # 보존력: 보존
    assert E_ncons[-1] - E_ncons[0] > 1.0                   # 비보존력: 에너지 유입


def test_static_vs_dynamic():
    assert correct("static-vs-dynamic") == [0, 1, 2]


def test_energy_drift_order():
    re = 2 ** (1 / 6)
    R0 = np.array([[0.0, 0, 0], [1.2, 0, 0]])
    V0 = np.array([[0.1, 0, 0], [-0.1, 0, 0]])
    E0 = 0.5 * np.sum(V0 ** 2) + lj_energy(R0)
    drifts = []
    for dt in [0.02, 0.005]:                                # 1/4로 줄이기
        E = verlet(R0.copy(), V0.copy(), lj_forces, dt, int(4.0 / dt))
        drifts.append(np.abs(E - E0).max())
    ratio = drifts[0] / drifts[1]
    assert 12 < ratio < 20                                  # Δt 1/4 → 표류 ~1/16 (2차)
    assert Q["energy-drift-order"]["answer"] == 16
