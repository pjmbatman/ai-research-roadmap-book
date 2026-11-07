"""9.5 퀴즈(content/09-numerical-methods/05-interpolation-and-quadrature/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import BarycentricInterpolator, CubicSpline

QUIZ = Path(__file__).parents[3] / "content/09-numerical-methods/05-interpolation-and-quadrature/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)


def test_runge_more_nodes():
    x_test = np.linspace(-1, 1, 2001)
    errs, where = [], []
    for n in (5, 10, 20):
        x = np.linspace(-1, 1, n + 1)
        e = np.abs(BarycentricInterpolator(x, runge(x))(x_test) - runge(x_test))
        errs.append(e.max())
        where.append(abs(x_test[e.argmax()]))
    assert errs[0] < errs[1] < errs[2]                 # 커진다
    assert min(where[1:]) > 0.8                        # n ≥ 10에서 최대 오차는 양 끝 근처
    assert correct("runge-more-nodes") == [1]


def test_extrapolation_guarantee():
    rng = np.random.default_rng(0)
    t = np.sort(np.concatenate([[0.0, 10.0], rng.uniform(0, 10, 38)]))
    sig = lambda t: np.exp(-0.2 * t) * np.sin(2 * t) + 0.3
    y = sig(t) + 0.01 * rng.normal(size=t.size)
    sp = CubicSpline(t, y)
    t_in = np.linspace(0, 10, 201)
    rms_in = np.sqrt(np.mean((sp(t_in) - sig(t_in)) ** 2))
    assert abs(sp(10.5) - sig(10.5)) > 10 * rms_in    # 구간 안 오차로 구간 밖을 말할 수 없다
    assert correct("extrapolation-guarantee") == [1]


def test_interpolate_noisy_data():
    rng = np.random.default_rng(1)
    x = np.linspace(-1, 1, 21)
    y = np.exp(x) + 0.05 * rng.normal(size=x.size)
    xf = np.linspace(-1, 1, 1001)
    p20 = np.polynomial.Chebyshev.fit(x, y, 20)
    p4 = np.polynomial.Chebyshev.fit(x, y, 4)
    assert np.max(np.abs(p20(x) - y)) < 1e-8          # 잔차 0
    assert np.max(np.abs(p20(xf) - np.exp(xf))) > 100
    assert np.max(np.abs(p4(xf) - np.exp(xf))) < 0.05
    # 기저를 바꿔도 같은 보간 다항식이다
    pm = np.polynomial.Polynomial.fit(x, y, 20)
    assert np.allclose(pm(xf[::50]), p20(xf[::50]), rtol=1e-4)
    assert correct("interpolate-noisy-data") == [1]


def test_simpson_halving():
    errs = []
    for n in (4, 8, 16):
        x = np.linspace(0, 1, n + 1)
        errs.append(abs(simpson(np.exp(x), x=x) - (np.e - 1)))
    ratios = [errs[0] / errs[1], errs[1] / errs[2]]
    q = Q["simpson-halving"]
    assert all(abs(r - q["answer"]) <= q["tolerance"] for r in ratios)


def test_gauss_exact_degree():
    xg, wg = np.polynomial.legendre.leggauss(3)
    exact = lambda k: (1 - (-1) ** (k + 1)) / (k + 1)
    ok = [k for k in range(10) if np.isclose(np.sum(wg * xg**k), exact(k), atol=1e-13)]
    max_exact = next(k for k in range(10) if k not in ok) - 1
    assert max_exact == Q["gauss-exact-degree"]["answer"] == 5


def test_sqrt_singularity_order():
    errs = []
    for n in (64, 128, 256, 512):
        x = np.linspace(0, 1, n + 1)
        errs.append(abs(simpson(np.sqrt(x), x=x) - 2 / 3))
    ratios = np.array(errs[:-1]) / np.array(errs[1:])
    assert np.allclose(ratios, 2**1.5, atol=0.1)
    assert correct("sqrt-singularity-order") == [1]


def test_monte_carlo_dimension():
    rng = np.random.default_rng(0)
    f = lambda x: np.cos(x.sum(axis=1))
    exact = lambda d: (((np.exp(1j) - 1) / 1j) ** d).real
    N = 4096
    # 1차원: 중점 규칙이 MC보다 훨씬 정확하다 (선택지 2가 틀림)
    g = ((np.arange(N) + 0.5) / N)[:, None]
    mc1 = [abs(f(rng.random((N, 1))).mean() - exact(1)) for _ in range(20)]
    assert abs(f(g).mean() - exact(1)) < 1e-3 * np.mean(mc1)
    # MC 오차 비율은 d와 무관하고, 표본 표준편차/√N으로 추정된다 (선택지 1, 3)
    for d in (2, 12):
        ests = np.array([f(rng.random((N, d))).mean() for _ in range(200)])
        se = f(rng.random((N, d))).std() / np.sqrt(N)
        assert 0.8 < ests.std() / se < 1.2
    assert correct("monte-carlo-dimension") == [0, 1, 3]
