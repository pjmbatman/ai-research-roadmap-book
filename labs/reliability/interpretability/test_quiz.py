"""21.5 퀴즈(content/21-reliability/05-interpretability/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/21-reliability/05-interpretability/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def perm_mse_increase(predict, X, y, cols):
    Xp = X.copy()
    Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
    return np.mean((predict(Xp) - y) ** 2) - np.mean((predict(X) - y) ** 2)


def test_permutation_correlated_split_and_refit():
    n = 4000
    x1 = rng.normal(size=n)
    x3 = x1 + 0.1 * rng.normal(size=n)
    y = 2 * x1 + 0.3 * rng.normal(size=n)
    X = np.column_stack([x1, x3])
    w = np.linalg.solve(X.T @ X + 10 * np.eye(2), X.T @ y)
    pred = lambda A: A @ w
    single = [perm_mse_increase(pred, X, y, [j]) for j in (0, 1)]
    both = perm_mse_increase(pred, X, y, [0, 1])
    assert both > sum(single)                                     # 함께 섞어야 묶음 전체가 보인다
    w3 = np.dot(x3, y) / (np.dot(x3, x3) + 10)                    # x1을 빼고 다시 학습
    assert np.mean((w3 * x3 - y) ** 2) - np.mean((pred(X) - y) ** 2) < 0.1
    assert correct("permutation-correlated") == [0]


def test_pdp_flat_but_ice_slopes_vary():
    z2 = rng.normal(size=5000)
    g = lambda z1: z1 * z2
    pdp_slope = (g(2.0).mean() - g(-2.0).mean()) / 4
    ice_slopes = (g(2.0) - g(-2.0)) / 4
    assert abs(pdp_slope) < 0.05 and np.std(ice_slopes) > 0.9
    assert correct("pdp-flat-interaction") == [0]


def test_gradxinput_linear():
    w, x = np.array([3.0, -2.0]), np.array([2.0, 1.0])
    attr = w * x
    assert attr[0] == Q["gradxinput-linear"]["answer"]
    assert np.isclose(attr.sum(), w @ x - 0.0)                    # 선형이면 완전성 성립


def test_ig_completeness():
    x, steps = 3.0, 1000
    alphas = (np.arange(steps) + 0.5) / steps
    ig = x * np.mean(2 * alphas * x)                              # f'(αx) = 2αx
    assert np.isclose(ig, Q["ig-completeness"]["answer"])
    assert np.isclose(x * 2 * x, 18.0)                            # gradient×input은 두 배


def test_downstream_sensor_intervention():
    n = 20000
    x1 = rng.normal(size=n)
    y = 2 * x1 + 0.5 * rng.normal(size=n)
    x3 = y + 0.1 * rng.normal(size=n)
    A = np.column_stack([x1, x3, np.ones(n)])
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    assert w[1] > 0.9                                             # 모델은 센서에 기댄다
    A_do = A.copy(); A_do[:, 1] += 1.0                            # 센서에 +1 개입: 실제 y는 그대로
    assert (A_do @ w).mean() - (A @ w).mean() > 0.9
    assert correct("attribution-not-causal") == [0]


def test_validation_methods():
    assert correct("validation-methods") == [0, 1, 2]
