"""19.5 퀴즈(content/19-generative-and-foundation-models/05-diffusion-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/19-generative-and-foundation-models/05-diffusion-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)
T = 1000
BETAS = np.linspace(1e-4, 0.02, T)
ABAR = np.cumprod(1 - BETAS)
PI, MU, SD = np.array([0.35, 0.65]), np.array([-1.5, 1.2]), np.array([0.25, 0.35])


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def sample_x0(n):
    k = (rng.random(n) < PI[1]).astype(int)
    return MU[k] + SD[k] * rng.normal(size=n)


def score(x, t):
    a = ABAR[t - 1]
    m, v = np.sqrt(a) * MU, a * SD**2 + 1 - a
    lw = np.log(PI) - 0.5 * (x[:, None] - m) ** 2 / v - 0.5 * np.log(v)
    r = np.exp(lw - lw.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    return (r * (-(x[:, None] - m) / v)).sum(1)


def test_forward_closed_form():
    a, x0 = 0.36, 2.0
    xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * rng.normal(size=400_000)
    assert abs(xt.mean() - Q["forward-closed-form"]["answer"]) < 0.01
    assert abs(xt.var() - 0.64) < 0.01


def test_forward_endpoint():
    x0 = sample_x0(100_000) * 3 + 5                            # 정규화하지 않은 데이터
    xT = np.sqrt(ABAR[-1]) * x0 + np.sqrt(1 - ABAR[-1]) * rng.normal(size=x0.size)
    assert ABAR[-1] < 1e-4 and abs(xT.mean()) < 0.05 and abs(xT.var() - 1) < 0.02
    assert correct("forward-endpoint") == [0]


def test_optimal_denoiser_and_eps_score_relation():
    t = 300
    a = ABAR[t - 1]
    x0, eps = sample_x0(300_000), rng.normal(size=300_000)
    xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
    e_star = -np.sqrt(1 - a) * score(xt, t)
    loss = lambda e_hat: np.mean((eps - e_hat) ** 2)
    assert loss(e_star) > 0.3                                   # 최적해에서도 0이 아니다
    assert loss(e_star) < min(loss(0.9 * e_star), loss(1.1 * e_star), loss(-e_star))
    # 구간별 E[eps | x_t] ≈ -sqrt(1 - abar) score
    bins = np.abs(xt - 0.8) < 0.02
    assert abs(eps[bins].mean() - (-np.sqrt(1 - a) * score(np.array([0.8]), t)[0])) < 0.03
    assert correct("optimal-denoiser") == [0]
    assert correct("eps-score-relation") == [0]


def test_steps_vs_quality_and_guidance():
    assert correct("steps-vs-quality") == [0]
    assert correct("guidance-weight") == [0]
    # guidance: 가우시안 두 성분에서 (1 + w) 외삽은 표준편차를 줄인다(정확한 score, DDIM 50 step)
    def run(w, n=20000, seed=3):
        g = np.random.default_rng(seed)
        ts = np.linspace(T, 1, 50).round().astype(int)
        x = g.normal(size=n)
        for i, t in enumerate(ts):
            a_t = ABAR[t - 1]
            a_p = ABAR[ts[i + 1] - 1] if i + 1 < len(ts) else 1.0
            m, v = np.sqrt(a_t) * MU[1], a_t * SD[1] ** 2 + 1 - a_t
            e_c = np.sqrt(1 - a_t) * (x - m) / v
            e_u = -np.sqrt(1 - a_t) * score(x, t)
            e = (1 + w) * e_c - w * e_u
            x0 = (x - np.sqrt(1 - a_t) * e) / np.sqrt(a_t)
            x = np.sqrt(a_p) * x0 + np.sqrt(1 - a_p) * e
        return x
    s0, s3 = run(0.0), run(3.0)
    assert s3.std() < s0.std() and s3.mean() > s0.mean()
