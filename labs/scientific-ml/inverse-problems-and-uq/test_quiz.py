"""24.7 퀴즈(content/24-scientific-ml/07-inverse-problems-and-uq/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/24-scientific-ml/07-inverse-problems-and-uq/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def blur(n=80):
    s = np.linspace(0, 1, n)
    A = np.exp(-0.5 * (s[:, None] - s[None, :]) ** 2 / 0.03**2)
    return A / A.sum(axis=1, keepdims=True), s


def test_ill_posed_noise_amplification():
    A, s = blur()
    x = (np.abs(s - 0.3) < 0.1).astype(float)
    eps = 0.01 * np.random.default_rng(0).normal(size=80)
    U, sv, Vt = np.linalg.svd(A)
    amplified = Vt.T @ ((U.T @ eps) / sv)                    # 잡음이 해에 남기는 성분
    assert np.linalg.norm(amplified) > 1e3 * np.linalg.norm(eps)
    assert correct("ill-posed-noise-amplification") == [0]


def test_lcurve_tradeoff():
    A, s = blur()
    y = A @ (np.abs(s - 0.3) < 0.1) + 0.01 * np.random.default_rng(0).normal(size=80)
    xs = [np.linalg.solve(A.T @ A + lam * np.eye(80), A.T @ y) for lam in (1e-6, 1e-4, 1e-2, 1.0)]
    res = [np.linalg.norm(A @ x - y) for x in xs]
    sol = [np.linalg.norm(x) for x in xs]
    assert np.all(np.diff(res) > 0) and np.all(np.diff(sol) < 0)
    assert correct("lcurve-tradeoff") == [0]


def test_tikhonov_prior_ratio():
    sigma, tau = 0.02, 0.1
    rng = np.random.default_rng(0)
    A, y = rng.normal(size=(10, 5)), rng.normal(size=10)
    mu_post = np.linalg.solve(A.T @ A / sigma**2 + np.eye(5) / tau**2, A.T @ y / sigma**2)
    lam = Q["tikhonov-prior-ratio"]["answer"]
    assert np.isclose(lam, sigma**2 / tau**2)
    assert np.allclose(mu_post, np.linalg.solve(A.T @ A + lam * np.eye(5), A.T @ y))


def test_sobol_ishigami_analytic():
    a, b = 7.0, 0.1
    V1 = 0.5 * (1 + b * np.pi**4 / 5) ** 2
    V2 = a**2 / 8
    V13 = b**2 * np.pi**8 * (1 / 18 - 1 / 50)
    V = V1 + V2 + V13
    assert np.isclose(V, a**2 / 8 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 18 + 0.5)
    S = np.array([V1, V2, 0.0]) / V
    ST = np.array([V1 + V13, V2, V13]) / V
    assert np.allclose(S, [0.314, 0.442, 0.0], atol=1e-3)     # 본문에 적은 해석해
    assert np.allclose(ST, [0.558, 0.442, 0.244], atol=1e-3)
    assert correct("sobol-interaction") == [0]


def test_local_vs_global_linear_case():
    rng = np.random.default_rng(0)
    g, sd = np.array([2.0, -1.0, 0.5]), np.array([1.0, 3.0, 2.0])   # 선형 모델의 기울기, 입력 표준편차
    X = rng.normal(size=(400000, 3)) * sd
    Y = X @ g
    S1 = [np.var(g[i] * X[:, i]) / np.var(Y) for i in range(3)]       # 선형·독립이면 Var(E[Y|X_i]) = (g_i sd_i)^2
    assert np.allclose(S1, (g * sd) ** 2 / np.sum((g * sd) ** 2), atol=0.01)
    assert correct("local-vs-global") == [0, 1, 2]


def test_jensen_propagation():
    theta = np.random.default_rng(0).normal(1.0, 0.5, size=2_000_000)
    assert np.isclose(np.mean(theta**2), Q["jensen-propagation"]["answer"], atol=0.005)
    assert correct("nonidentifiable-posterior") == [0]
