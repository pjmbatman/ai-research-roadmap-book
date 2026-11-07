"""24.5 퀴즈(content/24-scientific-ml/05-hybrid-and-surrogate-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/24-scientific-ml/05-hybrid-and-surrogate-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

R_J = 8.314


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def gp_mean(x_tr, y_tr, x_te, ell=15.0, sn=0.05):
    k = lambda a, b: np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)
    alpha = np.linalg.solve(k(x_tr, x_tr) + sn**2 * np.eye(x_tr.size), y_tr - y_tr.mean())
    return y_tr.mean() + k(x_te, x_tr) @ alpha


def test_residual_hybrid_extrapolation():
    ln_rate = lambda t, a, e: a - e / (R_J * t)
    truth, physics = (lambda t: ln_rate(t, 18.0, 60e3)), (lambda t: ln_rate(t, 16.0, 55e3))
    t_tr = np.linspace(300, 340, 15)
    resid = truth(t_tr) - physics(t_tr)
    t_far = np.array([400.0])
    hybrid = physics(t_far) + gp_mean(t_tr, resid, t_far)
    assert np.isclose(hybrid[0], physics(400.0) + resid.mean(), atol=1e-3)   # 선택지 0
    assert abs(hybrid[0] - truth(400.0)) > 0.3                                # 선택지 1(참값)이 아님
    assert correct("residual-hybrid-extrapolation") == [0]


def test_pod_modes():
    sv = np.array([10, 5, 2, 1, 0.5])
    frac = np.cumsum(sv**2) / np.sum(sv**2)
    r = int(np.argmax(frac >= 0.99)) + 1
    assert r == Q["pod-modes"]["answer"] == 3
    assert np.cumsum(sv)[2] / sv.sum() < 0.99                                # 특이값 비율로 세면 3개로 부족


def test_multifidelity_rho():
    x = np.linspace(0, 1, 101)
    f_hi = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    f_lo = 0.5 * f_hi + 10 * (x - 0.5) - 5
    rho = Q["multifidelity-rho"]["answer"]
    delta = f_hi - rho * f_lo
    assert np.allclose(np.diff(delta, 2), 0, atol=1e-9)                       # delta가 1차 함수
    assert not np.allclose(np.diff(f_hi - 0.5 * f_lo, 2), 0, atol=1e-6)


def test_maxvar_ignores_y():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    x_tr, x_c = np.array([[0.1], [0.5], [0.8]]), np.linspace(0, 1, 11)[:, None]
    stds = []
    for y in (np.array([0.0, 1.0, -1.0]), np.array([5.0, -3.0, 0.2])):       # 관측값이 전혀 다른 두 경우
        gp = GaussianProcessRegressor(RBF(0.1), alpha=1e-6, optimizer=None).fit(x_tr, y)
        stds.append(gp.predict(x_c, return_std=True)[1])
    assert np.allclose(stds[0], stds[1])                                       # 사후분산은 y와 무관
    assert correct("maxvar-ignores-y") == [0]


def test_choice_questions():
    assert correct("combination-choice") == [0, 1]
    assert correct("strongest-baseline") == [0]
