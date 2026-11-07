"""19.6 퀴즈(content/19-generative-and-foundation-models/06-flow-matching/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/19-generative-and-foundation-models/06-flow-matching/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)
PIS, MS, SS = np.array([0.3, 0.7]), np.array([-2.0, 2.0]), np.array([0.3, 0.3])


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def sample_data(n):
    k = rng.choice(2, size=n, p=PIS)
    return MS[k] + SS[k] * rng.normal(size=n)


def v_star(x, t):
    var = (1 - t) ** 2 + t**2 * SS[:, None] ** 2
    logw = np.log(PIS)[:, None] - 0.5 * (x - t * MS[:, None]) ** 2 / var - 0.5 * np.log(var)
    w = np.exp(logw - logw.max(0))
    w /= w.sum(0)
    return (w * (MS[:, None] + (t * SS[:, None] ** 2 - (1 - t)) / var * (x - t * MS[:, None]))).sum(0)


def test_cfm_loss_floor():
    n = 200_000
    x0, x1, t = rng.normal(size=n), sample_data(n), rng.uniform(size=n)
    xt, u = (1 - t) * x0 + t * x1, x1 - x0
    floor = np.mean((v_star(xt, t) - u) ** 2)
    assert 2.0 < floor < 2.4                                   # 참 속도장에서도 손실이 남는다
    assert np.mean((v_star(xt, t) + 0.1 - u) ** 2) > floor     # v*가 최소(조건부 기댓값)
    assert correct("cfm-loss-floor") == [1]


def test_monotone_coupling_has_no_floor():
    # 1차원에서 순서를 보존하는 짝(최적 수송)은 직선이 교차하지 않아 바닥이 0이다
    n = 50_000
    x0 = np.sort(rng.normal(size=n))
    x1 = np.sort(sample_data(n))
    t = rng.uniform(size=n)
    xt = (1 - t) * x0 + t * x1
    for tt in (0.3, 0.7):                                      # 같은 t에서 x_t도 순서를 보존한다
        assert np.all(np.diff((1 - tt) * x0 + tt * x1) >= 0)
    assert correct("reflow-tradeoff") == [0, 1, 2]


def test_velocity_from_eps():
    xt, eps, t = 1.0, 0.4, 0.5
    x1_hat = (xt - (1 - t) * eps) / t
    v = x1_hat - eps
    assert np.isclose(v, (x1_hat - xt) / (1 - t))
    assert np.isclose(v, (xt - eps) / t)
    assert abs(v - Q["velocity-from-eps"]["answer"]) <= Q["velocity-from-eps"]["tolerance"]


def test_solver_vs_model_error():
    # 참 속도장은 step을 늘리면 오차가 0으로 간다: 정규분포 → N(2, 0.5²)의 곧은 흐름은 Euler가 정확하고,
    # 굽은 흐름에서도 step을 늘리면 오차가 줄어든다
    x0 = rng.normal(size=2000)
    def euler(n):
        x = x0.copy()
        for k in range(n):
            x = x + v_star(x, k / n) / n
        return x
    ref = euler(2000)
    errs = [np.abs(np.sort(euler(n)) - np.sort(ref)).mean() for n in (1, 4, 32, 128)]
    assert errs[0] > errs[1] > errs[2] > errs[3] and errs[3] < 0.02
    assert correct("solver-vs-model-error") == [1]


def test_fm_vs_diffusion_and_loss_scale():
    # 직선 경로와 삼각 경로의 목표 속도 크기가 다르다 → 손실 값은 경로 사이에서 비교할 수 없다
    n = 100_000
    x1, eps, t = sample_data(n), rng.normal(size=n), rng.uniform(size=n)
    u_lin = x1 - eps
    u_tri = np.pi / 2 * (np.cos(np.pi / 2 * t) * x1 - np.sin(np.pi / 2 * t) * eps)
    assert abs(np.mean(u_tri**2) / np.mean(u_lin**2) - np.pi**2 / 8) < 0.05
    assert correct("fm-vs-diffusion") == [0, 1]
    assert correct("loss-across-paths") == [1]
