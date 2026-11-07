"""7.10 퀴즈(content/07-linear-algebra/10-symmetry-and-groups/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/07-linear-algebra/10-symmetry-and-groups/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def refl(theta):                                   # 각 theta 방향 직선에 대한 반사
    return rot(theta) @ np.diag([1.0, -1.0]) @ rot(-theta)


def test_which_are_groups():
    a, b = rng.uniform(0, 2 * np.pi, 2)
    assert np.allclose(rot(a) @ rot(b), rot(a + b))                      # 회전: 닫힘
    p, q = rng.permutation(5), rng.permutation(5)
    assert sorted(p[q]) == list(range(5))                                # 치환의 합성도 치환
    M = refl(a) @ refl(b)
    assert np.isclose(np.linalg.det(M), 1.0)                             # 반사 두 번 = 회전
    assert np.allclose(M, rot(2 * (a - b)))
    assert correct("which-are-groups") == [0, 1, 2]


def test_centroid_equivariant():
    X, R, t = rng.normal(size=(7, 2)), rot(0.4), np.array([1.0, -3.0])
    gX = X @ R.T + t
    assert np.allclose(gX.mean(0), R @ X.mean(0) + t)
    assert not np.allclose(gX.mean(0), X.mean(0))
    assert correct("centroid-equivariant") == [0]


def test_force_equivariant():
    # 쌍별 퍼텐셜 E = Σ (|x_i - x_j| - 1)^2 의 gradient로 힘을 계산하고 변환 규칙을 확인
    def energy_forces(X):
        d = X[:, None] - X[None, :]
        r = np.linalg.norm(d, axis=-1) + np.eye(len(X))
        E = 0.5 * np.sum((r - 1) ** 2 * (1 - np.eye(len(X))))
        F = -2 * np.sum(((r - 1) / r * (1 - np.eye(len(X))))[..., None] * d, axis=1)
        return E, F
    X, R = rng.normal(size=(4, 2)), rot(1.3)
    E1, F1 = energy_forces(X)
    E2, F2 = energy_forces(X @ R.T)
    assert np.isclose(E1, E2) and np.allclose(F2, F1 @ R.T)
    h, k = 1e-6, (0, 1)                                                   # 힘 = -dE/dx 수치 확인
    Xp, Xm = X.copy(), X.copy()
    Xp[k] += h
    Xm[k] -= h
    assert np.isclose(F1[k], -(energy_forces(Xp)[0] - energy_forces(Xm)[0]) / (2 * h), atol=1e-5)
    assert correct("force-equivariant") == [0]


def test_deepsets_parameters():
    d_in, d_out, n = 16, 32, 100
    assert 2 * d_in * d_out + d_out == Q["deepsets-parameters"]["answer"] == 1056
    assert (n * d_in) * (n * d_out) == 5_120_000


def test_augmentation_approximate():
    assert correct("augmentation-approximate") == [0]


def test_chirality_o3():
    X = rng.normal(size=(5, 2))
    mirror = X @ np.diag([1.0, -1.0])
    d = lambda Z: np.sort(np.linalg.norm(Z[:, None] - Z[None, :], axis=-1).ravel())
    assert np.allclose(d(X), d(mirror))                                   # 거리 특징은 거울상을 구별 못 함
    # 2차원에서 방향을 가진 넓이(외적 합)는 회전에 불변, 반사에 부호가 바뀐다
    signed_area = lambda Z: np.sum(Z[:-1, 0] * Z[1:, 1] - Z[1:, 0] * Z[:-1, 1])
    assert np.isclose(signed_area(X @ rot(0.7).T), signed_area(X))
    assert np.isclose(signed_area(mirror), -signed_area(X))
    assert correct("chirality-o3") == [0]


def test_noether_correspondence():
    def L_drift(k_ext):
        x = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.4, -0.9]])
        v = np.array([[0.0, 0.6], [-0.5, -0.2], [0.4, -0.3]])
        def force(x):
            f = -(x[:, None] - x[None, :]).sum(axis=1)
            f[:, 0] -= k_ext * x[:, 0]
            return f
        L0 = np.sum(x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0])
        drift = 0.0
        for _ in range(1000):
            vh = v + 0.005 * force(x)
            x = x + 0.01 * vh
            v = vh + 0.005 * force(x)
            drift = max(drift, abs(np.sum(x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) - L0))
        return drift
    assert L_drift(0.0) < 1e-12 and L_drift(0.5) > 0.1
    assert correct("noether-correspondence") == [0]
