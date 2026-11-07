"""14.1 퀴즈(content/14-information-and-learning-theory/01-entropy-and-kl/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

QUIZ = Path(__file__).parents[3] / "content/14-information-and-learning-theory/01-entropy-and-kl/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def kl(p, q):
    m = p > 0
    if np.any(q[m] == 0):
        return np.inf
    return np.sum(p[m] * np.log(p[m] / q[m]))


def normal_pdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def test_uniform_entropy_bits():
    p = np.full(16, 1 / 16)
    assert np.isclose(-np.sum(p * np.log2(p)), Q["uniform-entropy-bits"]["answer"])


def test_differential_entropy_facts():
    h = lambda s: 0.5 * np.log(2 * np.pi * np.e * s**2)
    assert h(0.002) < 0                                          # 음수 가능
    assert np.isclose(h(2.0) - h(0.002), np.log(1000))          # 단위에 따라 이동
    # KL은 단위 불변: 두 가우시안 KL의 닫힌 형태를 m와 mm로 계산
    kl_g = lambda m1, s1, m2, s2: np.log(s2 / s1) + (s1**2 + (m1 - m2) ** 2) / (2 * s2**2) - 0.5
    assert np.isclose(kl_g(0, 0.002, 0.001, 0.003), kl_g(0, 2.0, 1.0, 3.0))
    assert correct("differential-entropy-facts") == [0, 1]


def test_ce_equals_kl():
    p, q = np.array([0.7, 0.2, 0.1]), np.array([0.4, 0.4, 0.2])
    H = -np.sum(p * np.log(p))
    CE = -np.sum(p * np.log(q))
    assert np.isclose(CE, H + kl(p, q)) and H > 0
    assert not np.isclose(kl(p, q), kl(q, p))
    assert correct("ce-equals-kl") == [0]


def _fit(direction):
    x = np.linspace(-8, 8, 4001)
    dx = x[1] - x[0]
    p = 0.5 * normal_pdf(x, -2, 0.6) + 0.5 * normal_pdf(x, 2, 0.6)
    best = (np.inf, None, None)
    for mu in np.linspace(-3, 3, 61):
        for s in np.linspace(0.2, 3.0, 29):
            q = normal_pdf(x, mu, s)
            a, b = (p, q) if direction == "forward" else (q, p)
            v = np.sum(a * (np.log(a + 1e-300) - np.log(b + 1e-300))) * dx
            if v < best[0]:
                best = (v, mu, s)
    return best


def test_reverse_kl_mode():
    _, mu, s = _fit("reverse")
    assert abs(abs(mu) - 2) < 0.15 and s < 0.8                  # 한 봉우리에 맞춘 좁은 가우시안
    assert correct("reverse-kl-mode") == [0]


def test_mle_is_forward_kl():
    _, mu, s = _fit("forward")
    assert abs(mu) < 0.15 and s > 1.8                           # 두 봉우리를 덮는 넓은 가우시안
    assert correct("mle-is-forward-kl") == [0]


def test_disjoint_support():
    support = np.arange(10)
    p = np.zeros(10)
    p[[1, 2]] = 0.5
    w = []
    for shift in (2, 6):
        q = np.roll(p, shift)
        m = 0.5 * (p + q)
        assert kl(p, q) == np.inf
        assert np.isclose(0.5 * kl(p, m) + 0.5 * kl(q, m), np.log(2))
        w.append(wasserstein_distance(support, support, p, q))
    assert np.allclose(w, [2, 6])
    assert correct("disjoint-support") == [0]
