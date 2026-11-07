"""11.7 퀴즈(content/11-probability/07-stochastic-processes/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

QUIZ = Path(__file__).parents[3] / "content/11-probability/07-stochastic-processes/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_stationary_two_state():
    P = np.array([[0.9, 0.1], [0.3, 0.7]])
    p = np.array([0.0, 1.0])
    for _ in range(200):
        p = p @ P
    assert abs(p[0] - Q["stationary-two-state"]["answer"]) < 1e-9


def test_metropolis_normalizer():
    # 같은 난수로 Z가 다른 두 목표를 돌리면 체인이 완전히 같다
    def run(log_scale):
        r = np.random.default_rng(1)
        x, out = 0.0, []
        for _ in range(2000):
            prop = x + r.normal()
            if np.log(r.uniform()) < (-prop**2 / 2 + log_scale) - (-x**2 / 2 + log_scale):
                x = prop
            out.append(x)
        return np.array(out)
    assert np.array_equal(run(0.0), run(123.0))
    assert correct("metropolis-normalizer") == [0]


def test_periodic_chain():
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(np.array([0.5, 0.5]) @ P, [0.5, 0.5])
    p = np.array([1.0, 0.0])
    seq = [p := p @ P for _ in range(4)]
    assert np.allclose(seq[0], [0, 1]) and np.allclose(seq[1], [1, 0])
    assert np.allclose(sorted(np.linalg.eigvals(P)), [-1, 1])
    assert correct("periodic-chain") == [0]


def test_brownian_variance():
    dt = 0.01
    B = np.cumsum(rng.normal(scale=np.sqrt(dt), size=(20_000, 400)), axis=1)   # t = 0.01 ~ 4
    inc = B[:, 399] - B[:, 99]                                                  # B(4) - B(1)
    assert abs(inc.var() - Q["brownian-variance"]["answer"]) < 0.1


def test_ou_stationary_variance():
    theta, sigma, dt = 2.0, 1.0, 0.001
    x = np.zeros(5000)
    for _ in range(4000):                                   # t = 4, 상관시간 0.5의 8배
        x = x - theta * x * dt + sigma * np.sqrt(dt) * rng.normal(size=x.size)
    assert abs(sigma**2 / (2 * theta) - Q["ou-stationary-variance"]["answer"]) < 1e-12
    assert abs(x.var() - 0.25) < 0.015


def test_em_noise_scaling():
    T = 1.0
    for dt in (0.1, 0.01, 0.001):
        n = int(T / dt)
        w = (np.sqrt(dt) * rng.normal(size=(4000, n))).sum(axis=1)             # 정답: 분산 T 유지
        assert abs(w.var() - T) < 0.08
    assert correct("em-noise-scaling") == [0]


def test_random_walk_nonstationary():
    assert np.isclose(1 / (1 - 0.5**2), 4 / 3)
    e = rng.normal(size=(2, 1000, 500))
    y = lfilter([1.0], [1.0, -1.0], e, axis=-1)
    assert abs(y[0, :, -1].var() - 500) < 60                                    # Var y_t = t
    a = y[0] - y[0].mean(axis=1, keepdims=True)
    b = y[1] - y[1].mean(axis=1, keepdims=True)
    r = np.abs((a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1)))
    assert np.mean(r > 0.5) > 0.3
    assert correct("random-walk-nonstationary") == [0, 1, 2]
