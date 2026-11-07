"""19.3 퀴즈(content/19-generative-and-foundation-models/03-vae/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/19-generative-and-foundation-models/03-vae/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def linear_vae(x, w, b, sigma, mq, sq):
    r = x - b - w * mq
    rec = -0.5 * len(x) * np.log(2 * np.pi * sigma**2) - (r @ r + (w @ w) * sq**2) / (2 * sigma**2)
    kl = 0.5 * (mq**2 + sq**2 - 1 - np.log(sq**2))
    C = np.outer(w, w) + sigma**2 * np.eye(len(x))
    d = x - b
    log_px = -0.5 * (d @ np.linalg.solve(C, d) + np.log(np.linalg.det(2 * np.pi * C)))
    s2 = sigma**2 / (sigma**2 + w @ w)
    m = w @ d / (sigma**2 + w @ w)
    kl_post = 0.5 * (np.log(s2 / sq**2) + (sq**2 + (mq - m) ** 2) / s2 - 1)
    return rec, kl, log_px, kl_post


def test_elbo_gap_numeric():
    for _ in range(10):
        x, w, b = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
        rec, kl, log_px, kl_post = linear_vae(x, w, b, 0.6, rng.normal(), rng.uniform(0.2, 1.5))
        assert np.isclose(log_px - (rec - kl), kl_post)        # log p = ELBO + KL(q || 사후)
    q = Q["elbo-gap-numeric"]
    assert np.isclose(q["answer"], -2.5 - 0.4)


def test_beta_objective_bound():
    hits_above = 0
    for _ in range(200):
        x, w, b = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
        rec, kl, log_px, _ = linear_vae(x, w, b, rng.uniform(0.3, 2), 3 * rng.normal(), rng.uniform(0.05, 2))
        assert rec - 2.0 * kl <= rec - kl <= log_px + 1e-9          # beta > 1: 더 느슨한 하한
        hits_above += rec - 0.1 * kl > log_px                       # beta < 1: 하한이 깨질 수 있다
    assert hits_above > 0
    assert correct("beta-objective-bound") == [0, 1]


def test_reparam_sigma_grad():
    mu, s = 1.0, 0.5
    eps = rng.normal(size=1_000_000)
    g = np.mean(2 * (mu + s * eps) * eps)
    assert abs(g - Q["reparam-sigma-grad"]["answer"]) < 0.01
    assert Q["reparam-sigma-grad"]["answer"] == 2 * s


def test_reparam_why():
    assert correct("reparam-why") == [0]


def best_w(beta, sig2, v):
    ws = np.linspace(0.0, 3.0, 30001)
    a = ws / (ws**2 + beta * sig2)
    s2 = beta * sig2 / (ws**2 + beta * sig2)
    rec = -0.5 * np.log(2 * np.pi * sig2) - (v * (1 - ws * a) ** 2 + ws**2 * s2) / (2 * sig2)
    kl = 0.5 * (a**2 * v + s2 - 1 - np.log(s2))
    return ws[np.argmax(rec - beta * kl)]


def test_collapse_threshold():
    v, sig2 = 2.0, 0.5
    th = Q["collapse-threshold"]["answer"]
    assert best_w(th - 0.5, sig2, v) > 0.3
    assert best_w(th, sig2, v) == 0.0 and best_w(th + 1, sig2, v) == 0.0
    assert np.isclose(th, v / sig2)


def test_collapse_diagnosis():
    assert correct("collapse-diagnosis") == [0]
