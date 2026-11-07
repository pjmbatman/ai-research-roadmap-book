"""29.3 퀴즈(content/29-life-sciences/03-protein-structure/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/29-life-sciences/03-protein-structure/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def dist(X):
    return np.linalg.norm(X[:, None] - X[None], axis=-1)


def kabsch_rmsd(P, Q_, allow_reflection=False):
    P0, Q0 = P - P.mean(0), Q_ - Q_.mean(0)
    U, _, Vt = np.linalg.svd(P0.T @ Q0)
    d = 1.0 if allow_reflection else np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return np.sqrt(np.mean(np.sum((P0 @ R - Q0) ** 2, 1)))


def random_rotation():
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def test_mirror_image():
    X = rng.normal(size=(30, 3)) * 5
    M = X * np.array([1, 1, -1])
    assert np.allclose(dist(X), dist(M))
    assert kabsch_rmsd(M, X, allow_reflection=True) < 1e-8
    assert kabsch_rmsd(M, X) > 0.5
    assert correct("mirror-image") == [1]


def test_invariant_quantities():
    X = rng.normal(size=(30, 3)) * 5
    R = random_rotation()
    Y = X @ R.T + np.array([3.0, -2.0, 7.0])
    P = X + rng.normal(0, 0.3, X.shape)
    assert np.allclose(dist(X), dist(Y))                                   # 거리 행렬
    assert np.array_equal(dist(X) < 8, dist(Y) < 8)                        # 접촉 지도
    assert np.isclose(kabsch_rmsd(P, X), kabsch_rmsd(P, Y))                # Kabsch RMSD
    plain = lambda A, B: np.sqrt(np.mean(np.sum((A - B) ** 2, 1)))
    assert not np.isclose(plain(P, X), plain(P, Y))                        # 정렬 없는 RMSD
    assert correct("invariant-quantities") == [0, 1, 2]


def test_tm_score_d0():
    d0 = 1.24 * (100 - 15) ** (1 / 3) - 1.8
    assert abs(d0 - Q["tm-score-d0"]["answer"]) <= Q["tm-score-d0"]["tolerance"]
    assert np.isclose(1 / (1 + 1), 0.5) and np.isclose(1 / (1 + 4), 0.2)  # d=d0, d=2d0의 기여


def test_domain_hinge_rmsd_vs_local():
    """도메인 각각이 정확해도 배치가 틀리면 RMSD는 크고 도메인별 RMSD는 0이다."""
    A, B = rng.normal(size=(20, 3)) * 3, rng.normal(size=(20, 3)) * 3 + np.array([15.0, 0, 0])
    X = np.vstack([A, B])
    th = np.deg2rad(40)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    P = np.vstack([A, (B - A.mean(0)) @ Rz.T + A.mean(0)])
    assert kabsch_rmsd(P, X) > 1.5                                         # 배치 오차가 전체 RMSD를 키운다
    assert kabsch_rmsd(P[20:], X[20:]) < 1e-8 and kabsch_rmsd(P[:20], X[:20]) < 1e-8
    assert correct("domain-hinge") == [1]


def test_low_confidence_and_predicted():
    assert correct("low-confidence-region") == [2]
    assert correct("predicted-vs-experimental") == [0]
