"""7.8 퀴즈(content/07-linear-algebra/08-manifolds-and-geometry/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.linalg import expm, null_space
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

QUIZ = Path(__file__).parents[3] / "content/07-linear-algebra/08-manifolds-and-geometry/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def sphere(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def sphere_jac(th, ph):
    return np.array([[np.cos(th) * np.cos(ph), -np.sin(th) * np.sin(ph)],
                     [np.cos(th) * np.sin(ph), np.sin(th) * np.cos(ph)],
                     [-np.sin(th), 0.0]])


def test_so3_dimension():
    # 항등원에서 R^T R = I를 미분하면 A^T + A = 0: 접공간 = 반대칭 행렬, 차원은 선형 조건의 null space로 센다
    L = np.zeros((9, 9))
    for k in range(9):
        E = np.zeros(9)
        E[k] = 1.0
        A = E.reshape(3, 3)
        L[:, k] = (A + A.T).ravel()
    assert null_space(L).shape[1] == Q["so3-dimension"]["answer"] == 3
    # 반대칭 행렬의 지수는 회전이다
    W = rng.normal(size=(3, 3))
    R = expm(W - W.T)
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)


def test_chart_singularity():
    assert np.linalg.matrix_rank(sphere_jac(1.0, 0.5)) == 2
    J0 = sphere_jac(0.0, 0.5)
    assert np.linalg.matrix_rank(J0) == 1 and np.allclose(J0[:, 1], 0.0)     # 해석적으로도 φ 열이 0
    # 축을 돌린 chart (x축을 극으로)를 쓰면 북극(0, 0, 1)에서도 rank 2
    rot = lambda th, ph: np.array([sphere(th, ph)[1], sphere(th, ph)[2], sphere(th, ph)[0]])
    th, ph = np.pi / 2, 0.0                                                    # rot(th, ph) = (0, 0, 1)
    assert np.allclose(rot(th, ph), [0, 0, 1])
    Jr = sphere_jac(th, ph)[[1, 2, 0]]
    assert np.linalg.matrix_rank(Jr) == 2
    assert correct("chart-singularity") == [0]


def test_jacobian_tangent_plane():
    th, ph = 1.0, 0.5
    J, p = sphere_jac(th, ph), sphere(th, ph)
    assert np.allclose(J.T @ p, 0.0)                     # 열공간 ⟂ 법선 p
    assert null_space(J).shape[1] == 0                   # null space는 {0}
    assert np.linalg.matrix_rank(J) == 2                 # 접공간 차원 2
    assert correct("jacobian-tangent-plane") == [0]


def test_curve_rank_vs_dimension():
    t = np.linspace(-1, 1, 200)
    C = np.stack([t, t**2, t**3], axis=1)
    C = C - C.mean(axis=0)
    assert np.linalg.matrix_rank(C) == 3
    assert all(np.linalg.matrix_rank(np.array([[1.0], [2 * s], [3 * s * s]])) == 1 for s in t)
    assert correct("curve-rank-vs-dimension") == [0]


def test_lerp_midpoint_norm():
    p, q = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    v = np.linalg.norm((p + q) / 2)
    assert abs(v - Q["lerp-midpoint-norm"]["answer"]) <= Q["lerp-midpoint-norm"]["tolerance"]


def test_latent_dim_not_dof():
    # 자유도 2인 데이터(2개 파라미터로 생성)를 32차원에 선형으로 심으면 rank는 2: 32는 상한일 뿐
    z = rng.normal(size=(500, 2))
    X = np.tanh(z @ rng.normal(size=(2, 32)))
    lin = z @ rng.normal(size=(2, 32))
    assert np.linalg.matrix_rank(lin) == 2
    assert X.shape[1] == 32
    assert correct("latent-dim-not-dof") == [0]


def test_pca_on_swiss_roll():
    X, _ = make_swiss_roll(n_samples=1500, random_state=0)
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False) ** 2
    ratio = s / s.sum()
    assert np.allclose(ratio, [0.41, 0.31, 0.29], atol=0.01)
    _, idx = NearestNeighbors(n_neighbors=12).fit(X).kneighbors(X)
    loc = []
    for nb in idx[::10]:
        v = np.linalg.svd(X[nb] - X[nb].mean(axis=0), compute_uv=False) ** 2
        loc.append(v / v.sum())
    assert np.mean(loc, axis=0)[2] < 0.01                # 국소적으로는 2차원
    assert correct("pca-on-swiss-roll") == [0]
