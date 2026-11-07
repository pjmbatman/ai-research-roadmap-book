"""13.4 퀴즈(content/13-bayesian-inference/04-latent-variables-and-em/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import logsumexp

QUIZ = Path(__file__).parents[3] / "content/13-bayesian-inference/04-latent-variables-and-em/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def em_1d(x, pi, mu, sd, n_iter):
    lls = []
    for _ in range(n_iter):
        lj = np.log(pi) + stats.norm.logpdf(x[:, None], mu, sd)
        lls.append(logsumexp(lj, axis=1).sum())
        r = np.exp(lj - logsumexp(lj, axis=1, keepdims=True))
        Nk = r.sum(0)
        pi, mu = Nk / len(x), (r * x[:, None]).sum(0) / Nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / Nk)
    return mu, np.array(lls)


def test_responsibility_value():
    w = np.array([0.2, 0.8]) * stats.norm.pdf(1.0, [0.0, 2.0], 1.0)
    q = Q["responsibility-value"]
    assert abs(w[0] / w.sum() - q["answer"]) <= q["tolerance"]


def test_marginal_likelihood_form():
    x = rng.normal(size=50)
    pi, mu, sd = np.array([0.4, 0.6]), np.array([-1.0, 1.0]), np.array([1.0, 0.5])
    dens = pi * stats.norm.pdf(x[:, None], mu, sd)
    true_ll = np.log(dens.sum(1)).sum()
    # 몬테카를로: z를 뽑아 x를 만드는 생성 과정의 밀도와 같은지 (점 하나에서)
    assert np.isclose(true_ll, logsumexp(np.log(pi) + stats.norm.logpdf(x[:, None], mu, sd), axis=1).sum())
    assert not np.isclose(true_ll, np.log(dens).sum())                                   # 로그와 합 순서 바꿈
    assert not np.isclose(true_ll, np.log(dens.max(1)).sum())                            # hard assignment
    assert correct("marginal-likelihood-form") == [0]


def test_m_step_mean():
    x, r = np.array([0.0, 2.0, 4.0]), np.array([1.0, 0.5, 0.0])
    q = Q["m-step-mean"]
    assert abs(np.sum(r * x) / r.sum() - q["answer"]) <= q["tolerance"]


def test_em_monotone():
    x = np.concatenate([rng.normal(1.0, 0.5, 150), rng.normal(3.0, 0.8, 350)])
    _, lls = em_1d(x, np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([1.0, 1.0]), 100)
    assert np.all(np.diff(lls) > -1e-9)
    assert correct("em-monotone") == [0]


def test_em_local_optimum():
    r1 = np.random.default_rng(1)
    x3 = np.concatenate([r1.normal(-3, 0.6, 150), r1.normal(0, 0.6, 150), r1.normal(0.9, 0.6, 150)])
    finals = []
    for seed in range(10):
        r = np.random.default_rng(seed)
        _, lls = em_1d(x3, np.full(3, 1 / 3), r.choice(x3, 3, replace=False), np.full(3, 1.0), 300)
        finals.append(lls[-1])
    assert max(finals) - min(finals) > 1.0                                  # 서로 다른 해
    # 퇴화: 한 성분이 점 하나에 붙고 분산이 0으로 가면 로그우도가 커진다
    x = np.array([0.0, 1.0, 2.0, 5.0])
    ll = [np.log(0.5 * stats.norm.pdf(x, 1.0, 1.0) + 0.5 * stats.norm.pdf(x, 5.0, s)).sum() for s in (1e-1, 1e-3, 1e-6)]
    assert ll[0] < ll[1] < ll[2]
    assert correct("em-local-optimum") == [0, 1, 2]


def test_missing_mechanism():
    r = np.random.default_rng(2)
    n = 20000
    u = r.normal(size=n)
    y = 2.0 + u + 0.5 * r.normal(size=n)
    obs = y < 2.8
    b1, b0 = np.polyfit(u[obs], y[obs], 1)
    assert abs(np.mean(b0 + b1 * u) - 2.0) > 0.1                            # 회귀로 채워도 bias
    assert abs(y[obs].mean() - 2.0) > 0.3
    assert correct("missing-mechanism") == [0]


def test_vae_elbo():
    # ELBO = log p - KL: q가 사후와 다르면 ELBO < log p
    pi, mu, sd, x = np.array([0.5, 0.5]), np.array([0.0, 2.0]), np.array([1.0, 1.0]), np.array([0.3])
    lj = np.log(pi) + stats.norm.logpdf(x[:, None], mu, sd)
    q = np.array([[0.5, 0.5]])
    elbo = np.sum(q * (lj - np.log(q)))
    assert elbo < logsumexp(lj, axis=1).sum()
    assert correct("vae-elbo") == [0]
