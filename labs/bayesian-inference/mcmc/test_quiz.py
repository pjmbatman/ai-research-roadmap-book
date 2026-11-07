"""13.5 퀴즈(content/13-bayesian-inference/05-mcmc/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/13-bayesian-inference/05-mcmc/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def mh_chain(log_p, step, n, rng, th0=(3.0, -3.0)):
    th = np.array(th0)
    lp, out, acc = log_p(th), np.empty((n, 2)), 0
    for t in range(n):
        prop = th + step * rng.normal(size=2)
        lp1 = log_p(prop)
        if np.log(rng.random()) < lp1 - lp:
            th, lp, acc = prop, lp1, acc + 1
        out[t] = th
    return out, acc / n


P = np.linalg.inv(np.array([[1.0, 0.95], [0.95, 1.0]]))
log_p = lambda th: -0.5 * th @ P @ th


def test_normalizing_constant_cancels():
    rng = np.random.default_rng(0)
    a, _ = mh_chain(log_p, 0.5, 3000, rng)
    b, _ = mh_chain(lambda th: log_p(th) + 123.4, 0.5, 3000, np.random.default_rng(0))   # log Z만 다르다
    assert np.array_equal(a, b)
    assert correct("normalizing-constant-cancels") == [0]


def test_hastings_correction():
    alpha = min(1.0, (2 * 0.3) / (8 * 0.1))
    assert abs(alpha - Q["hastings-correction"]["answer"]) < 1e-12


def test_high_acceptance_is_not_efficiency():
    rng = np.random.default_rng(0)
    small, a_small = mh_chain(log_p, 0.05, 5000, rng)
    mid, a_mid = mh_chain(log_p, 0.5, 5000, rng)
    assert a_small > 0.85 and a_mid < a_small
    assert np.abs(small[1000:].mean(0)).max() > np.abs(mid[1000:].mean(0)).max()   # 작은 보폭은 아직 치우쳐 있다
    assert correct("acceptance-rate-misread") == [1]


def test_gibbs_lag1_is_rho_squared():
    rho, rng = 0.9, np.random.default_rng(1)
    x1, x2, xs = 0.0, 0.0, []
    for _ in range(40000):
        x1 = rho * x2 + np.sqrt(1 - rho**2) * rng.normal()
        x2 = rho * x1 + np.sqrt(1 - rho**2) * rng.normal()
        xs.append(x1)
    x = np.array(xs[1000:]) - np.mean(xs[1000:])
    lag1 = (x[:-1] @ x[1:]) / (x @ x)
    assert abs(lag1 - Q["gibbs-autocorrelation"]["answer"]) < Q["gibbs-autocorrelation"]["tolerance"]


def test_leapfrog_stability_limit():
    omega_max = np.sqrt(np.linalg.eigvalsh(P).max())
    assert 0.3 * omega_max < 2 < 0.5 * omega_max
    assert correct("hmc-divergence") == [0]


def test_convergence_statements():
    assert correct("convergence-statements") == [0, 1]


def test_mcse():
    assert abs(2.0 / np.sqrt(100) - Q["mcse-from-ess"]["answer"]) < 1e-12
