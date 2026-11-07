"""16.2 퀴즈(content/16-classical-ml/02-linear-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

QUIZ = Path(__file__).parents[3] / "content/16-classical-ml/02-linear-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_ridge_prior_lambda():
    sigma, tau = 0.5, 0.25
    lam = sigma**2 / tau**2
    assert np.isclose(lam, Q["ridge-prior-lambda"]["answer"])
    # MAP 해와 Ridge 해가 같은지 확인: 음의 로그 사후분포의 gradient가 0
    X, y = rng.normal(size=(30, 3)), rng.normal(size=30)
    w = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y)
    grad = -X.T @ (y - X @ w) / sigma**2 + w / tau**2
    assert np.allclose(grad, 0)


def test_lasso_exact_zeros():
    X, y = rng.normal(size=(100, 5)), rng.normal(size=100)
    X[:, 0] *= 3
    y += 2 * X[:, 0]
    assert np.any(Lasso(alpha=0.1).fit(X, y).coef_ == 0.0)
    assert np.all(Ridge(alpha=1e4).fit(X, y).coef_ != 0.0)
    assert correct("lasso-exact-zeros") == [0]


def test_standardize_before_penalty():
    raw = load_diabetes(scaled=False).data
    y = load_diabetes().target
    raw2 = raw.copy()
    raw2[:, 3] *= 0.1333
    sel = lambda Xr: tuple(Lasso(alpha=10.0).fit(Xr, y).coef_ != 0)
    assert sel(raw) != sel(raw2)                                    # 표준화 없으면 단위에 따라 선택이 바뀐다
    p1 = make_pipeline(StandardScaler(), Lasso(alpha=10.0)).fit(raw, y).predict(raw)
    p2 = make_pipeline(StandardScaler(), Lasso(alpha=10.0)).fit(raw2, y).predict(raw2)
    assert np.allclose(p1, p2)
    assert correct("standardize-before-penalty") == [0]


def test_threshold_from_costs():
    c_fp, c_fn = 1.0, 9.0
    p = np.linspace(0, 1, 100001)
    cost_alarm, cost_none = (1 - p) * c_fp, p * c_fn
    thr = p[np.argmax(cost_alarm < cost_none)]                      # 경보가 나아지는 첫 확률
    assert np.isclose(thr, Q["threshold-from-costs"]["answer"], atol=1e-4)


def test_lasso_dropped_feature():
    assert correct("lasso-dropped-feature") == [1]


def test_polynomial_extrapolation():
    f = lambda t: np.exp(-t) * np.cos(4 * t)
    r = np.random.default_rng(0)
    t = np.sort(r.uniform(0, 2, 25))[:, None]
    y = f(t[:, 0]) + r.normal(0, 0.05, 25)
    m = make_pipeline(StandardScaler(), PolynomialFeatures(12), LinearRegression()).fit(t, y)
    t_out = np.linspace(2, 2.5, 50)[:, None]
    assert np.sqrt(np.mean((m.predict(t_out) - f(t_out[:, 0])) ** 2)) > 100
    assert correct("polynomial-extrapolation") == [1]
