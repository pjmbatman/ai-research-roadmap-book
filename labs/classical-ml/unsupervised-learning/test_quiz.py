"""16.5 퀴즈(content/16-classical-ml/05-unsupervised-learning/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import adjusted_rand_score, silhouette_score

QUIZ = Path(__file__).parents[3] / "content/16-classical-ml/05-unsupervised-learning/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_pca_uncorrelated_not_independent():
    t = np.linspace(0, 8, 2000)
    S = np.c_[np.sin(2 * np.pi * t), np.sign(np.sin(2 * np.pi * 0.3 * t))]
    X = S @ np.array([[1.0, 0.6], [0.5, 1.0]]).T
    match = lambda Z: np.abs(np.corrcoef(S.T, Z.T)[:2, 2:]).max(axis=1)
    Zp = PCA(2).fit_transform(X)
    assert abs(np.corrcoef(Zp.T)[0, 1]) < 1e-8                  # PCA 성분은 무상관
    assert match(Zp).min() < 0.9                                 # 그러나 원천과 일치하지 않음
    assert match(FastICA(2, random_state=0).fit_transform(X)).min() > 0.99
    # 무상관 성분을 회전해도 무상관이다(백색화한 뒤 회전)
    W = Zp / Zp.std(0)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert abs(np.corrcoef((W @ R).T)[0, 1]) < 1e-8
    assert correct("pca-uncorrelated-not-independent") == [0]


def test_units_change_clusters():
    r = np.random.default_rng(3)
    label = np.repeat([0, 1, 2], 100)
    F = np.c_[np.array([300.0, 310.0, 320.0])[label] + r.normal(0, 2, 300), 1e5 + r.normal(0, 3000, 300)]
    ari = lambda Fx: adjusted_rand_score(label, KMeans(3, n_init=10, random_state=0).fit_predict(Fx))
    assert ari(F) < 0.05 and ari(F / [1, 1000]) > 0.9
    assert correct("units-change-clusters") == [0]


def test_silhouette_on_uniform():
    D = rng.uniform(0, 4, (300, 2))
    sil = [silhouette_score(D, KMeans(k, n_init=10, random_state=0).fit_predict(D)) for k in range(2, 7)]
    assert min(sil) > 0.3 and max(sil) - min(sil) < 0.1          # 구조가 없어도 양수, 곡선은 평평
    assert correct("silhouette-on-uniform") == [0]


def test_mahalanobis_anticorrelated():
    rho = 0.95
    S = np.array([[1, rho], [rho, 1]])
    x = np.array([1.0, -1.0])
    d2 = x @ np.linalg.solve(S, x)
    assert np.isclose(d2, 2 / (1 - rho))
    assert abs(d2 - Q["mahalanobis-anticorrelated"]["answer"]) <= Q["mahalanobis-anticorrelated"]["tolerance"]


def test_tsne_what_to_read():
    assert correct("tsne-what-to-read") == [0, 1]


def test_cluster_validation():
    # 군집 내 제곱합은 k에 대해 단조 감소한다(선택지 2가 근거가 될 수 없는 이유)
    D = rng.uniform(0, 4, (200, 2))
    inertia = [KMeans(k, n_init=10, random_state=0).fit(D).inertia_ for k in range(1, 7)]
    assert np.all(np.diff(inertia) < 0)
    assert correct("cluster-validation") == [0, 1]
