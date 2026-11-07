"""10.3 퀴즈(content/10-optimization/03-first-order-methods/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/10-optimization/03-first-order-methods/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def gd_final(lam, lr, steps=2000):
    x = np.ones_like(lam)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            x = x - lr * lam * x
    return np.linalg.norm(x)


def test_max_stable_lr_value():
    lam = np.array([2.0, 50.0])
    ans = Q["max-stable-lr-value"]["answer"]
    assert np.isclose(ans, 2 / lam.max())
    assert gd_final(lam, 0.99 * ans) < 1e-6
    big = gd_final(lam, 1.01 * ans)
    assert not np.isfinite(big) or big > 1e6


def test_slowest_direction():
    lam = np.array([1.0, 100.0])
    x = np.ones(2)
    for _ in range(50):
        x = x - (1 / lam.max()) * lam * x
    assert abs(x[0]) > 0.5 and abs(x[1]) < 1e-12           # 작은 곡률 방향만 남는다
    assert correct("slowest-direction") == [0]


def test_batch_size_noise():
    g = rng.normal(size=200000)                              # 샘플별 gradient (스칼라)
    std = {}
    for B in (10, 40):
        idx = rng.choice(g.size, size=(5000, B))
        std[B] = g[idx].mean(axis=1).std()
    assert abs(std[40] / std[10] - Q["batch-size-noise"]["answer"]) < 0.03


def test_sgd_plateau():
    X = rng.normal(size=(500, 2))
    y = X @ np.array([1.0, -2.0]) + 0.5 * rng.normal(size=500)
    w_star = np.linalg.lstsq(X, y, rcond=None)[0]
    loss = lambda w: 0.5 * np.mean((X @ w - y) ** 2)
    floors = {}
    for lr in (0.1, 0.01):
        r = np.random.default_rng(1)
        w, hist = np.zeros(2), []
        for k in range(20000):
            i = r.integers(500)
            w = w - lr * X[i] * (X[i] @ w - y[i])
            hist.append(loss(w) - loss(w_star))
        floors[lr] = np.mean(hist[-5000:])
    assert 5 < floors[0.1] / floors[0.01] < 20                # 바닥이 학습률에 대략 비례
    assert correct("sgd-plateau") == [0]


def test_armijo_meaning():
    f = lambda x: x**4
    x, c = 1.5, 1e-4
    g = 4 * x**3
    t = 1.0
    while f(x - t * g) > f(x) - c * t * g * g:
        t /= 2
    assert f(x - 1.0 * g) > f(x)                              # 큰 step은 f를 키운다
    ts = np.linspace(0, t * 4, 4001)
    t_exact = ts[np.argmin(f(x - ts * g))]
    assert not np.isclose(t, t_exact, rtol=0.05)              # 받아들인 step은 최적 step이 아니다
    assert f(x - t * g) <= f(x) - c * t * g * g
    assert correct("armijo-meaning") == [0]


def hb_steps(lam, lr, beta, tol=1e-6, max_steps=100000):
    x, v = np.ones_like(lam), np.zeros_like(lam)
    for k in range(1, max_steps):
        v = beta * v - lr * lam * x
        x = x + v
        if np.linalg.norm(x) < tol:
            return k


def test_momentum_effects():
    ks = {}
    for kappa in (100, 10000):
        lam = np.array([1.0, kappa])
        r = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
        ks[kappa] = (hb_steps(lam, 2 / (1 + kappa), 0.0), hb_steps(lam, 4 / (np.sqrt(kappa) + 1) ** 2, r**2))
    assert 70 < ks[10000][0] / ks[100][0] < 130               # GD: kappa에 비례
    assert 7 < ks[10000][1] / ks[100][1] < 13                 # HB: sqrt(kappa)에 비례
    lam = np.array([1.0, 100.0])
    assert gd_final(lam, 0.99 * 2 / 100) < 1e-3               # beta = 0 경계
    # beta = 0.9이면 2/L보다 큰 학습률에서도 안정하다
    x, v = np.ones(2), np.zeros(2)
    for _ in range(4000):
        v = 0.9 * v - 0.035 * lam * x
        x = x + v
    assert np.linalg.norm(x) < 1e-3 and 0.035 > 2 / 100
    assert correct("momentum-effects") == [0, 1]


def test_convex_rate():
    # 볼록이지만 강볼록이 아닌 f(x) = x^4 (|x| <= 1에서 L = 12): 오차가 O(1/k) 수준으로 느리게 준다
    x, vals = 1.0, []
    for _ in range(20000):
        x = x - (1 / 12) * 4 * x**3
        vals.append(x**4)
    vals = np.array(vals)
    ratio = vals[1:] / vals[:-1]
    assert ratio[-1] > 0.999                                   # 선형 수렴이 아니다(축소율이 1에 다가감)
    k = np.arange(1, vals.size + 1)
    assert np.all(vals <= 12 * 1.0 / (2 * k) + 1e-12)          # L |x0 - x*|^2 / (2k) 보장
    assert correct("convex-rate") == [0]
