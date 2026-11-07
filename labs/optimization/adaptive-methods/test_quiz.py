"""10.4 퀴즈(content/10-optimization/04-adaptive-methods/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/10-optimization/04-adaptive-methods/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def adam_dir(g, m, v, t, b1=0.9, b2=0.999, eps=1e-8, bc=True):
    m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
    mh, vh = (m / (1 - b1**t), v / (1 - b2**t)) if bc else (m, v)
    return mh / (np.sqrt(vh) + eps), m, v


def test_adam_first_step():
    u, _, _ = adam_dir(np.array([0.02, -50.0]), np.zeros(2), np.zeros(2), 1)
    step = 0.001 * np.abs(u)
    assert np.allclose(step, Q["adam-first-step"]["answer"], rtol=1e-4)


def test_bias_correction_why():
    u, _, _ = adam_dir(np.array([1.0]), np.zeros(1), np.zeros(1), 1, bc=False)
    assert np.isclose(u[0], 0.1 / np.sqrt(0.001), rtol=1e-6)          # 약 3.16배
    # t가 크면 보정 계수는 1로 간다
    assert abs(1 / (1 - 0.999**10000) - 1) < 1e-4
    assert correct("bias-correction-why") == [0]


def steps_to(opt, H, lr, max_steps=3000):
    x, m, v = np.array([0.8, -0.5]), np.zeros(2), np.zeros(2)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, max_steps + 1):
            g = H @ x
            if opt == "momentum":
                m = 0.9 * m + g
                x = x - lr * m
            else:                                                   # rmsprop
                v = 0.999 * v + 0.001 * g * g
                x = x - lr * g / (np.sqrt(v) + 1e-8)
            if not np.isfinite(x).all():
                return np.inf
            if 0.5 * x @ H @ x < 1e-6:
                return k
    return np.inf


def test_diagonal_preconditioning():
    lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
    c = s = np.sqrt(0.5)
    R = np.array([[c, -s], [s, c]])
    H0 = np.diag([1.0, 100.0])
    H1 = R @ H0 @ R.T
    assert np.allclose(np.linalg.eigvalsh(H1), [1, 100])               # 회전해도 조건수는 같다
    best = {(o, n): min(steps_to(o, H, lr) for lr in lrs) for o in ("momentum", "rmsprop") for n, H in (("axis", H0), ("rot", H1))}
    assert best["rmsprop", "axis"] * 10 < best["momentum", "axis"]
    assert best["rmsprop", "rot"] > best["momentum", "rot"]
    assert abs(best["momentum", "axis"] - best["momentum", "rot"]) < 0.3 * best["momentum", "axis"]
    assert correct("diagonal-preconditioning") == [0]


def run_decay(decoupled, a, lam, lr=0.01, steps=20000):
    w, m, v = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in range(1, steps + 1):
        g = a * (w - 1) + (0 if decoupled else lam * w)
        u, m, v = adam_dir(g, m, v, t)
        w = w - lr * u - (lr * lam * w if decoupled else 0)
    return w


def test_adam_l2_vs_adamw():
    a, lam = np.array([0.01, 1.0, 100.0]), 1.0
    w_l2, w_w = run_decay(False, a, lam), run_decay(True, a, lam)
    assert np.allclose(w_l2, a / (a + lam), atol=1e-3)                # 곡률(gradient)이 큰 좌표는 덜 감쇠
    assert np.ptp(w_w) < 1e-3                                         # AdamW는 같은 비율
    assert correct("adam-l2-vs-adamw") == [0]


def test_decay_transfer():
    a, lam, lr = np.array([0.5, 2.0]), 0.3, 0.05
    w1 = w2 = np.zeros(2)
    for _ in range(500):
        w1 = w1 - lr * (a * (w1 - 1) + lam * w1)
        w2 = (1 - lr * lam) * w2 - lr * a * (w2 - 1)
    assert np.allclose(w1, w2)                                        # SGD: 같다
    # decoupled 감쇠의 세기는 학습률에 비례한다
    x = np.ones(1)
    for lr_ in (0.01, 0.02):
        assert np.isclose((x * (1 - lr_ * lam))[0], 1 - lr_ * lam)
    assert correct("decay-transfer") == [0, 1]


def test_warmup_reason():
    assert correct("warmup-reason") == [0]


def test_accumulation_divisor():
    X, y, w = rng.normal(size=(64, 3)), rng.normal(size=64), rng.normal(size=3)
    grad = lambda Xb, yb: Xb.T @ (Xb @ w - yb) / len(yb)
    total = sum(grad(X[i:i + 16], y[i:i + 16]) for i in range(0, 64, 16))
    assert np.allclose(total / Q["accumulation-divisor"]["answer"], grad(X, y))
