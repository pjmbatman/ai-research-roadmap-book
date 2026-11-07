"""11.6 퀴즈(content/11-probability/06-sampling/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy import integrate, stats

QUIZ = Path(__file__).parents[3] / "content/11-probability/06-sampling/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_mc_error_scaling():
    sigma = 1.3
    n1 = (sigma / 0.02) ** 2
    n2 = (sigma / 0.002) ** 2
    assert np.isclose(n2 / n1, Q["mc-error-scaling"]["answer"])


def test_mc_dimension():
    # 단위초입방체 [0,1]^d 위 f(x) = Σx_i / d 의 표준오차는 d가 커져도 늘지 않는다
    for d in (1, 10, 100):
        fx = rng.uniform(size=(20_000, d)).mean(axis=1)
        assert fx.std() <= np.sqrt(1 / 12) + 1e-3
    assert correct("mc-dimension") == [0]


def test_inverse_cdf_exponential():
    lam = 2.0
    u = rng.uniform(size=200_000)
    candidates = [-np.log(1 - u) / lam, -lam * np.log(1 - u), np.log(1 - u) / lam, 1 - np.exp(-lam * u)]
    ok = [i for i, x in enumerate(candidates) if stats.kstest(x, stats.expon(scale=1 / lam).cdf).pvalue > 1e-3]
    assert ok == correct("inverse-cdf-exponential") == [0]


def test_rejection_acceptance():
    M = 2.5
    x, u = rng.uniform(size=400_000), rng.uniform(size=400_000)
    rate = np.mean(u * M < stats.beta(2, 5).pdf(x))
    assert abs(rate - Q["rejection-acceptance"]["answer"]) < 0.005


def test_ess_weights():
    w = np.array([1.0, 1.0, 2.0])
    ess = w.sum() ** 2 / np.sum(w**2)
    assert abs(ess - Q["ess-weights"]["answer"]) <= Q["ess-weights"]["tolerance"]
    assert np.isclose((5 * w).sum() ** 2 / np.sum((5 * w) ** 2), ess)    # 척도에 무관


def test_is_proposal_tails():
    # E_q[w^2] = ∫ p^2 / q : 유한하면 가중치 분산이 유한하다
    def second_moment(loc, scale):
        val, _ = integrate.quad(lambda x: stats.norm.pdf(x) ** 2 / stats.norm.pdf(x, loc, scale), -12, 12, limit=200)
        return val
    assert second_moment(0, 2.0) < 2                                       # 유한, ESS/N ≈ 0.66
    assert second_moment(2, 1.0) > 50                                      # 위치가 어긋나 e^4 ≈ 54.6
    assert second_moment(0, 0.5) > 1e6                                     # 적분 구간을 넓힐수록 발산
    assert correct("is-proposal-tails") == [0]


def test_reparam_vs_score():
    mu, N, R = 1.0, 100, 4000
    x = mu + rng.normal(size=(R, N))
    rep = (2 * x).mean(axis=1)
    sf = (x**2 * (x - mu)).mean(axis=1)
    sfb = ((x**2 - (mu**2 + 1)) * (x - mu)).mean(axis=1)
    for g in (rep, sf, sfb):
        assert abs(g.mean() - 2 * mu) < 0.05                               # 모두 불편
    assert rep.var() < sfb.var() < sf.var()
    assert correct("reparam-vs-score") == [0, 1, 2]
