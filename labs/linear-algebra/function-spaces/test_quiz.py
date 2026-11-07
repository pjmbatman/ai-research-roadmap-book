"""7.9 퀴즈(content/07-linear-algebra/09-function-spaces/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.integrate import quad

QUIZ = Path(__file__).parents[3] / "content/07-linear-algebra/09-function-spaces/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def within(qid, value):
    q = Q[qid]
    return abs(value - q["answer"]) <= q["tolerance"]


def test_l2_norm_sin():
    v, _ = quad(lambda x: np.sin(x) ** 2, 0, 2 * np.pi)
    assert within("l2-norm-sin", v)
    assert abs(quad(lambda x: np.sin(x) * np.cos(x), 0, 2 * np.pi)[0]) < 1e-12


def test_grid_inner_product_scaling():
    f, g = np.exp, lambda x: np.sin(np.pi * x)
    dot = lambda N: f((np.arange(N) + 0.5) / N) @ g((np.arange(N) + 0.5) / N)
    assert 9.9 < dot(1000) / dot(100) < 10.1
    assert correct("grid-inner-product-scaling") == [0]


def test_monomial_gram():
    xq, wq = np.polynomial.legendre.leggauss(40)
    gram = lambda B: (B * wq) @ B.T
    c_mono = np.linalg.cond(gram(np.vander(xq, 12, increasing=True).T))
    c_leg = np.linalg.cond(gram(np.polynomial.legendre.legvander(xq, 11).T))
    assert 1e7 < c_mono < 1e8 and np.isclose(c_leg, 23.0)
    # 두 basis는 같은 공간: 단항식 계수 ↔ Legendre 계수가 가역 변환
    M = np.polynomial.legendre.legvander(xq, 11)
    V = np.vander(xq, 12, increasing=True)
    T, *_ = np.linalg.lstsq(M, V, rcond=None)
    assert np.allclose(M @ T, V) and np.linalg.matrix_rank(T) == 12
    assert correct("monomial-gram") == [0]


def test_fd_error_order():
    def err(N):
        x = 2 * np.pi * np.arange(N) / N
        h = 2 * np.pi / N
        u = np.exp(np.sin(x))
        return np.abs((np.roll(u, -1) - np.roll(u, 1)) / (2 * h) - np.cos(x) * u).max()
    for N in [32, 64]:
        assert within("fd-error-order", err(N) / err(2 * N))


def test_derivative_unbounded():
    def lam_max(N):
        h = 2 * np.pi / N
        I = np.eye(N)
        return np.linalg.eigvalsh((2 * I - np.roll(I, 1, 1) - np.roll(I, -1, 1)) / h**2).max()
    assert np.isclose(lam_max(64) / lam_max(32), 4.0)
    assert np.isclose(lam_max(64), 4 / (2 * np.pi / 64) ** 2)
    assert correct("derivative-unbounded") == [0]


def test_valid_kernels():
    kernels = [
        lambda a, b: np.exp(-(a - b) ** 2 / 2),
        lambda a, b: (a * b + 1) ** 2,
        lambda a, b: a * b,
        lambda a, b: -((a - b) ** 2),
    ]
    psd = []
    for k in kernels:
        ok = True
        for _ in range(50):
            x = rng.normal(size=6) * 2
            G = k(x[:, None], x[None, :])
            ok &= np.linalg.eigvalsh(G).min() > -1e-9
        psd.append(ok)
    assert [i for i, ok in enumerate(psd) if ok] == correct("valid-kernels") == [0, 1, 2]


def test_representer_theorem():
    # 해 f = Σ α_i k(·, x_i)는 Gram 행렬 (N, N)만으로 정해지고, 계수는 N개
    x = np.sort(rng.uniform(0, 1, 50))
    y = np.sin(2 * np.pi * x)
    K = np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * 0.1**2))
    alpha = np.linalg.solve(K + 0.1 * np.eye(50), y)
    assert alpha.shape == (50,)
    assert correct("representer-theorem") == [0]
