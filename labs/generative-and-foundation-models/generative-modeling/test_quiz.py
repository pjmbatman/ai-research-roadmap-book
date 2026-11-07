"""19.1 퀴즈(content/19-generative-and-foundation-models/01-generative-modeling/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

QUIZ = Path(__file__).parents[3] / "content/19-generative-and-foundation-models/01-generative-modeling/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)
K, SD = 8, 0.15
ang = 2 * np.pi * np.arange(K) / K
CENTERS = 2.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def sample(n):
    c = rng.integers(K, size=n)
    return CENTERS[c] + SD * rng.normal(size=(n, 2)), c


def log_gauss(x, mu, sd):
    return -((x - mu) ** 2).sum(-1) / (2 * sd**2) - np.log(2 * np.pi * sd**2)


def test_mle_is_forward_kl():
    # 가우시안 하나의 NLL - H(p_data) = KL(p_data || p_theta) >= 0 (MC 추정)
    x, _ = sample(20000)
    log_pdata = logsumexp(log_gauss(x[:, None], CENTERS[None], SD), axis=1) - np.log(K)
    mu, S = x.mean(0), np.cov(x.T)
    d = x - mu
    log_q = -0.5 * np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d) - 0.5 * np.log(np.linalg.det(2 * np.pi * S))
    assert np.mean(log_pdata - log_q) > 0
    assert correct("mle-is-forward-kl") == [0]


def test_conditional_nll_gap():
    x, c = sample(20000)
    lp = log_gauss(x[:, None], CENTERS[None], SD)
    gap = -(logsumexp(lp, axis=1) - np.log(K)).mean() + lp[np.arange(len(x)), c].mean()
    q = Q["conditional-nll-gap"]
    assert abs(gap - q["answer"]) <= q["tolerance"]
    assert np.isclose(q["answer"], np.log(8), atol=1e-3)


def test_regression_mean_not_sample():
    x, _ = sample(20000)
    y = x[np.abs(x[:, 0]) < 0.3, 1]
    assert abs(y.mean()) < 0.2 and np.mean(np.abs(y - y.mean()) < 0.5) == 0
    assert correct("regression-mean-not-sample") == [0]


def test_exact_likelihood_families():
    assert correct("exact-likelihood-families") == [0, 1]


def test_kde_likelihood_unreliable():
    x_te, _ = sample(2000)
    true = (logsumexp(log_gauss(x_te[:, None], CENTERS[None], SD), axis=1) - np.log(K)).mean()
    g, _ = sample(200)                                          # 참 분포와 같은 샘플러
    kde = lambda h: (logsumexp(log_gauss(x_te[:, None], g[None], h), axis=1) - np.log(len(g))).mean()
    vals = [kde(h) for h in [0.02, 0.08, 0.15, 0.4]]
    assert all(v < true for v in vals)                          # 모두 참값보다 낮다
    assert vals[0] < true - 3                                   # 작은 h는 크게 나빠진다
    assert correct("kde-likelihood-unreliable") == [0]


def test_copier_detection():
    x_tr, _ = sample(2000)
    x_ho, _ = sample(1000)
    copies = x_tr[rng.choice(len(x_tr), 1000)]
    nn = lambda s: np.sqrt(((s[:, None] - x_tr[None]) ** 2).sum(-1)).min(1)
    assert np.all(nn(copies) == 0) and np.median(nn(x_ho)) > 0.01
    assert correct("copier-detection") == [0]
