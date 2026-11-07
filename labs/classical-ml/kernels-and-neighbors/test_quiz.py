"""16.3 퀴즈(content/16-classical-ml/03-kernels-and-neighbors/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

QUIZ = Path(__file__).parents[3] / "content/16-classical-ml/03-kernels-and-neighbors/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_knn_scale_units():
    rng = np.random.default_rng(0)
    c, t = rng.uniform(0.01, 0.2, 600), rng.uniform(280, 360, 600)
    y = 50 * c**2 * np.exp(-2000 * (1 / t - 1 / 320)) + rng.normal(0, 0.02, 600)
    X = np.column_stack([c, t])
    raw = KNeighborsRegressor(10).fit(X[:400], y[:400]).score(X[400:], y[400:])
    std = make_pipeline(StandardScaler(), KNeighborsRegressor(10)).fit(X[:400], y[:400]).score(X[400:], y[400:])
    assert raw < 0.3 < 0.95 < std
    assert correct("knn-scale-units") == [0]


def test_knn_irrelevant_features():
    def task(d_noise, r):
        X = r.uniform(-1, 1, (500, 2 + d_noise))
        return X, np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + r.normal(0, 0.1, 500)

    scores = {}
    for d in (0, 64):
        r = np.random.default_rng(1)
        (Xa, ya), (Xb, yb) = task(d, r), task(d, r)
        scores[d] = (KNeighborsRegressor(10, algorithm="kd_tree").fit(Xa, ya).score(Xb, yb),
                     RidgeCV().fit(Xa, ya).score(Xb, yb))
    assert scores[0][0] - scores[64][0] > 0.5                       # kNN은 크게 나빠진다
    assert abs(scores[0][1] - scores[64][1]) < 0.1                  # 선형 모델은 거의 그대로
    assert correct("knn-irrelevant-features") == [0]


def test_non_support_vector_removal():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal([-1.5, 0], 0.7, (100, 2)), rng.normal([1.5, 0], 0.7, (100, 2))])
    y = np.repeat([0, 1], 100)
    a = SVC(kernel="linear", C=1.0, tol=1e-6).fit(X, y)
    keep = np.ones(200, bool)
    keep[np.setdiff1d(np.arange(200), a.support_)[:50]] = False
    b = SVC(kernel="linear", C=1.0, tol=1e-6).fit(X[keep], y[keep])
    assert np.allclose(a.coef_, b.coef_, atol=1e-4) and np.allclose(a.intercept_, b.intercept_, atol=1e-4)
    assert correct("non-support-vector-removal") == [0]


def test_krr_equals_gp_mean():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 30))[:, None]
    y = np.sin(x[:, 0]) + rng.normal(0, 0.1, 30)
    xt = np.linspace(-2, 12, 141)[:, None]
    krr = KernelRidge(kernel="rbf", gamma=0.5, alpha=0.01).fit(x, y).predict(xt)
    mean, std = GaussianProcessRegressor(RBF(1.0), alpha=0.01, optimizer=None).fit(x, y).predict(xt, return_std=True)
    assert np.allclose(krr, mean, atol=1e-8)
    assert std[np.argmin(np.abs(xt[:, 0] - 5))] < 0.1 < 0.9 < std[-1]   # 데이터에서 멀수록 분산이 크다
    assert abs(mean[-1]) < 0.15                                        # 사전평균 0으로 돌아간다
    assert correct("krr-equals-gp-mean") == [0, 1]


def test_length_scale_choice():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 30))[:, None]
    y = np.sin(x[:, 0]) + rng.normal(0, 0.1, 30)
    mid = ((x[1:, 0] + x[:-1, 0]) / 2)[:, None]                        # 이웃한 학습 점의 중간
    pred_short = KernelRidge(kernel="rbf", gamma=1 / (2 * 0.1**2), alpha=0.01).fit(x, y).predict(mid)
    pred_good = KernelRidge(kernel="rbf", gamma=0.5, alpha=0.01).fit(x, y).predict(mid)
    assert np.mean(np.abs(pred_short)) < np.mean(np.abs(pred_good))   # 짧은 ℓ은 점 사이에서 0 쪽으로 꺼진다
    assert np.isclose(np.mean(np.diff(x[:, 0])), 0.33, atol=0.05)
    assert correct("length-scale-choice") == [0]


def test_kernel_cost_scaling():
    assert (8000 / 2000) ** 3 == Q["kernel-cost-scaling"]["answer"]
