"""19.4 퀴즈(content/19-generative-and-foundation-models/04-gan-and-flows/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

QUIZ = Path(__file__).parents[3] / "content/19-generative-and-foundation-models/04-gan-and-flows/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
x = np.linspace(-12, 16, 7001)
dx = x[1] - x[0]
log_pd = logsumexp([norm.logpdf(x, -2, 0.5), norm.logpdf(x, 2, 0.5)], axis=0) - np.log(2)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def js(lp, lq):
    lm = np.logaddexp(lp, lq) - np.log(2)
    return 0.5 * np.sum(np.exp(lp) * (lp - lm) + np.exp(lq) * (lq - lm)) * dx


def test_optimal_discriminator():
    assert np.isclose(Q["optimal-discriminator"]["answer"], 0.3 / (0.3 + 0.1))


def test_gan_vanishing_gradient():
    v = lambda mu: 2 * js(log_pd, norm.logpdf(x, mu, 0.5)) - np.log(4)
    assert abs(v(8.0) - 0.0) < 1e-6                              # JS = log 2로 포화
    assert abs(v(8.01) - v(7.99)) / 0.02 < 1e-5                  # 기울기 소실
    assert abs(v(3.01) - v(2.99)) / 0.02 > 0.1
    assert correct("gan-vanishing-gradient") == [0]


def test_mode_collapse_cause():
    mus, sds = np.linspace(-3, 3, 61), np.linspace(0.2, 3.0, 29)
    vals = np.array([[js(log_pd, norm.logpdf(x, m, s)) for s in sds] for m in mus])
    i, j = np.unravel_index(np.argmin(vals), vals.shape)
    assert abs(mus[i]) < 1e-9 and sds[j] > 1.5                    # JS 최적해는 두 모드를 덮는다
    one_mode = js(log_pd, norm.logpdf(x, -2, 0.5))
    assert 0 < one_mode - vals[i, j] < 0.03                       # 차이가 작다
    assert correct("mode-collapse-cause") == [0]


def test_gan_loss_not_progress():
    assert np.isclose(2 * np.log(2), 1.386, atol=1e-3)
    assert correct("gan-loss-not-progress") == [0]


def test_flow_log_density():
    q = Q["flow-log-density"]
    assert abs(norm.logpdf(0.0) - np.log(3) - q["answer"]) <= q["tolerance"]
    assert abs(norm.logpdf(1.0, 1.0, 3.0) - q["answer"]) <= q["tolerance"]


def test_coupling_cheap_logdet():
    rng = np.random.default_rng(0)
    s = lambda a: np.tanh(a @ np.array([0.7, -0.3]))[:, None] * np.array([0.5, -0.2])
    t = lambda a: (a**2) @ np.array([[0.3, 0.1], [-0.2, 0.4]])
    f = lambda v: np.concatenate([v[:, :2], v[:, 2:] * np.exp(s(v[:, :2])) + t(v[:, :2])], 1)
    v = rng.normal(size=(1, 4))
    J = np.stack([(f(v + e) - f(v - e))[0] / 2e-6 for e in np.eye(4) * 1e-6], 1)
    assert np.isclose(np.log(abs(np.linalg.det(J))), s(v[:, :2]).sum(), atol=1e-6)
    assert correct("coupling-cheap-logdet") == [0]
