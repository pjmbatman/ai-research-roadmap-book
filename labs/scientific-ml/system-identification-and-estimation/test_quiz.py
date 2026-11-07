"""24.6 퀴즈(content/24-scientific-ml/06-system-identification-and-estimation/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/24-scientific-ml/06-system-identification-and-estimation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def sens(f, theta, t, h=1e-6):
    cols = []
    for j in range(len(theta)):
        up, dn = theta.copy(), theta.copy()
        up[j] *= np.exp(h)
        dn[j] *= np.exp(-h)
        cols.append((f(t, *up) - f(t, *dn)) / (2 * h))
    return np.column_stack(cols)


def test_structural_vs_practical():
    v_two = lambda t, v, tau: v * (1 - np.exp(-t / tau))
    corr = []
    for t in (np.linspace(0.02, 1, 50), np.linspace(0.2, 10, 50)):
        S = sens(v_two, np.array([39.24, 4.0]), t)
        assert np.linalg.matrix_rank(S) == 2                     # 두 구간 모두 구조적으로는 식별 가능
        C = np.linalg.inv(S.T @ S)
        corr.append(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]))
    assert corr[0] > 0.999 and corr[1] < 0.95                    # 짧은 구간만 강하게 상관
    assert correct("structural-vs-practical") == [0]


def test_observability_rank():
    dt = 0.1
    F = np.array([[1, dt], [0, 1]])
    rank = lambda H: np.linalg.matrix_rank(np.vstack([H, H @ F]))
    assert rank(np.array([[0.0, 1.0]])) == Q["observability-rank"]["answer"] == 1
    assert rank(np.array([[1.0, 0.0]])) == 2


def test_threshold_units():
    # 같은 물리 관계 dT/dt = -0.5 (T/1 K) - 0.02 (T/1 K)^2 를 mK로 쓰면 계수가 각각 1e-3, 1e-6배가 된다
    T = np.linspace(1, 3, 50)
    lib = lambda T: np.column_stack([T, T**2])
    dT = -0.5 * T - 0.02 * T**2
    coef_k = np.linalg.lstsq(lib(T), dT, rcond=None)[0]
    coef_mk = np.linalg.lstsq(lib(1e3 * T), 1e3 * dT, rcond=None)[0]
    assert np.allclose(coef_k, [-0.5, -0.02])
    assert np.allclose(coef_mk, [-0.5, -0.02e-3])                # T^2 항만 1000배 작아져 같은 임계값에서 잘린다
    assert correct("threshold-units") == [0]


def test_noisy_derivative_amplification():
    rng = np.random.default_rng(0)
    sigma = 0.1
    for dt in (0.01, 0.001):
        noise = sigma * rng.normal(size=200001)
        d = (noise[2:] - noise[:-2]) / (2 * dt)
        assert np.isclose(d.std(), sigma / (np.sqrt(2) * dt), rtol=0.02)
    assert correct("noisy-sindy") == [0, 1]


def test_kalman_gain_limits():
    P, H = np.array([[2.0, 0.3], [0.3, 1.0]]), np.array([[1.0, 0.0]])
    for R, expect_zero in [(1e12, True), (1e-12, False)]:
        K = P @ H.T / (H @ P @ H.T + R)
        assert np.allclose(K, 0, atol=1e-9) == expect_zero
    assert correct("kalman-gain-limits") == [0]


def test_effective_sample_size():
    w = np.array([0.5, 0.5, 0.0, 0.0])
    assert 1 / np.sum(w**2) == Q["effective-sample-size"]["answer"] == 2
