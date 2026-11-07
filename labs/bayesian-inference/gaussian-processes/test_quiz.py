"""13.7 퀴즈(content/13-bayesian-inference/07-gaussian-processes/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/13-bayesian-inference/07-gaussian-processes/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def rbf(a, b, ell=1.0, sf=1.0):
    return sf**2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)


def posterior(xtr, ytr, xte, ell, sf, sn):
    K = rbf(xtr, xtr, ell, sf) + sn**2 * np.eye(xtr.size)
    Ks = rbf(xtr, xte, ell, sf)
    mean = Ks.T @ np.linalg.solve(K, ytr)
    var = sf**2 - np.sum(Ks * np.linalg.solve(K, Ks), axis=0)
    return mean, var


def crossings(ell, rng, n=300):
    x = np.linspace(0, 10, 401)
    L = np.linalg.cholesky(rbf(x, x, ell) + 1e-6 * np.eye(x.size))
    f = L @ rng.normal(size=(x.size, n))
    return np.mean(np.sum(np.diff(np.sign(f), axis=0) != 0, axis=0))


def test_lengthscale_crossings():
    rng = np.random.default_rng(0)
    ratio = crossings(0.25, rng) / crossings(1.0, rng)
    assert 3.6 < ratio < 4.4
    assert correct("lengthscale-crossings") == [0]


def test_prior_sd_far_away():
    xtr = np.linspace(0, 5, 10)
    _, var = posterior(xtr, np.sin(xtr), np.array([100.0]), 1.0, 2.0, 0.1)
    assert abs(np.sqrt(var[0]) - Q["prior-sd-far-away"]["answer"]) < 1e-6


def test_posterior_var_independent_of_y():
    xtr, xte = np.linspace(0, 5, 10), np.linspace(-1, 6, 20)
    _, v1 = posterior(xtr, np.sin(xtr), xte, 1.0, 1.0, 0.1)
    _, v2 = posterior(xtr, 10 * np.cos(3 * xtr), xte, 1.0, 1.0, 0.1)
    assert np.allclose(v1, v2)
    assert correct("posterior-var-independent-of-y") == [0]


def test_single_point_variance():
    _, var = posterior(np.array([0.0]), np.array([0.3]), np.array([0.0]), 1.0, 1.0, 1.0)
    assert abs(var[0] - Q["single-point-variance"]["answer"]) < 1e-12


def test_lml_short_lengthscale():
    rng = np.random.default_rng(2)
    f = lambda x: np.sin(x) + 0.3 * x
    x = np.sort(rng.uniform(0, 10, 30))
    y = f(x) - f(x).mean() + 0.1 * rng.normal(size=30)

    def terms(ell):
        L = np.linalg.cholesky(rbf(x, x, ell) + 0.01 * np.eye(30))
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return -0.5 * y @ a, -np.sum(np.log(np.diag(L)))

    fit_s, cx_s = terms(0.05)
    fit_m, cx_m = terms(1.0)
    assert abs(fit_s - fit_m) < 2 and cx_m - cx_s > 30        # 차이는 복잡도 항에서 온다
    assert correct("lml-short-lengthscale") == [0]


def test_isotropic_scaling():
    # 길이척도 7e4에서 온도 100 K 차이는 상관 1과 구별되지 않는다
    assert np.exp(-0.5 * (100 / 7e4) ** 2) > 0.999
    assert correct("isotropic-scaling") == [0]


def test_cubic_cost():
    assert (8000 / 2000) ** 3 == Q["cubic-cost"]["answer"]
