"""13.2 퀴즈(content/13-bayesian-inference/02-conjugate-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy import stats

QUIZ = Path(__file__).parents[3] / "content/13-bayesian-inference/02-conjugate-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_gamma_poisson_posterior():
    counts, t = np.array([2, 8]), np.array([1.0, 3.0])
    rate = np.linspace(1e-4, 10, 200001)
    logpost = stats.gamma.logpdf(rate, 2, scale=1.0) + stats.poisson.logpmf(counts[:, None], rate * t[:, None]).sum(0)
    p = np.exp(logpost - logpost.max())
    mean = np.sum(rate * p) / p.sum()
    q = Q["gamma-poisson-posterior"]
    assert abs(mean - q["answer"]) <= q["tolerance"]
    assert abs(counts.sum() / t.sum() - q["answer"]) > 0.05                 # MLE 2.5와 다르다


def test_conjugacy_meaning():
    # Beta 사전 × Bernoulli 우도 = Beta: 격자에서 확인
    theta = np.linspace(1e-6, 1 - 1e-6, 10001)
    post = stats.beta.pdf(theta, 3, 2) * theta**4 * (1 - theta) ** 1
    post /= post.sum() * (theta[1] - theta[0])
    assert np.max(np.abs(post - stats.beta.pdf(theta, 7, 3))) < 1e-3
    assert correct("conjugacy-meaning") == [0]


def test_normal_normal_weight():
    sigma, tau0, n, ybar = 0.3, 0.25, 9, 0.5
    b = np.linspace(-2, 2, 400001)
    logpost = stats.norm.logpdf(ybar, b, sigma / np.sqrt(n)) + stats.norm.logpdf(b, 0, tau0)
    p = np.exp(logpost - logpost.max())
    mean = np.sum(b * p) / p.sum()
    q = Q["normal-normal-weight"]
    assert abs(mean - q["answer"]) <= q["tolerance"]


def blr(Phi, y, alpha, sigma):
    S = np.linalg.inv(alpha * np.eye(Phi.shape[1]) + Phi.T @ Phi / sigma**2)
    return S @ Phi.T @ y / sigma**2, S


def test_predictive_variance_parts():
    feats = lambda x: np.vander(np.asarray(x) / 3, 6, increasing=True)
    sigma = 0.1
    for n_per in (15, 1500):
        x = np.r_[rng.uniform(-3, -1, n_per), rng.uniform(1.5, 3, n_per)]
        y = np.sin(x) + sigma * rng.normal(size=x.size)
        m, S = blr(feats(x), y, 1.0, sigma)
        P = feats([-2.0, 0.25, 4.0])
        epi = np.einsum("nd,de,ne->n", P, S, P)
        assert np.all(sigma**2 + epi >= sigma**2)                          # σ 아래로 내려가지 않는다
        assert epi[0] < epi[1] < epi[2]                                    # 데이터 없는 곳에서 커진다
    assert correct("predictive-variance-parts") == [0, 1]


def test_fixed_noise_overconfident():
    feats = lambda x: np.vander(np.asarray(x) / 3, 6, increasing=True)
    x = np.r_[rng.uniform(-3, -1, 15), rng.uniform(1.5, 3, 15)]
    y = np.sin(x) + 0.2 * rng.normal(size=30)
    _, S_true = blr(feats(x), y, 1.0, 0.2)
    _, S_small = blr(feats(x), y, 1.0, 0.05)
    P = feats([0.25])
    assert (P @ S_small @ P.T)[0, 0] < (P @ S_true @ P.T)[0, 0]             # 가중치 불확실성도 줄어든다
    assert correct("fixed-noise-overconfident") == [0]


def test_evidence_occam():
    r = np.random.default_rng(3)
    x = r.uniform(-1, 1, 25)
    y = 1.0 - 2.0 * x + 1.5 * x**3 + 0.2 * r.normal(size=25)
    le, rmse = [], []
    for deg in range(10):
        Phi = np.vander(x, deg + 1, increasing=True)
        C = 0.2**2 * np.eye(25) + Phi @ Phi.T / 0.1
        le.append(stats.multivariate_normal(np.zeros(25), C).logpdf(y))
        rmse.append(np.sqrt(np.mean((Phi @ np.linalg.lstsq(Phi, y, rcond=None)[0] - y) ** 2)))
    assert np.argmax(le) == 3 and np.all(np.diff(le[3:]) < 0)
    assert np.all(np.diff(rmse) <= 1e-12)                                  # 학습 오차는 계속 준다
    assert correct("evidence-occam") == [0]
