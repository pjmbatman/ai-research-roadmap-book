"""24.4 퀴즈(content/24-scientific-ml/04-neural-operators/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/24-scientific-ml/04-neural-operators/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def sample(n, K=12, decay=1.5):
    k = np.arange(1, K + 1)
    return (rng.normal(size=(n, K)) + 1j * rng.normal(size=(n, K))) / k**decay


def on_grid(coef, N):
    k = np.arange(1, coef.shape[1] + 1)
    return 2 * np.real(coef @ np.exp(2j * np.pi * np.outer(k, np.arange(N) / N)))


def spectral_conv(u, w):
    N, K = u.shape[1], w.size
    u_hat = np.fft.rfft(u, axis=1) / N
    out = np.zeros_like(u_hat)
    out[:, 1:K + 1] = w * u_hat[:, 1:K + 1]
    return np.fft.irfft(out * N, n=N, axis=1)


def fit(u_in, u_out, K):
    N = u_in.shape[1]
    a, b = np.fft.rfft(u_in, axis=1)[:, 1:K + 1] / N, np.fft.rfft(u_out, axis=1)[:, 1:K + 1] / N
    return (b * a.conj()).sum(0) / (np.abs(a) ** 2).sum(0)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_fno_resolution():
    f, w = sample(3), rng.normal(size=16) + 1j * rng.normal(size=16)
    stencil = lambda u: 0.25 * np.roll(u, 1, 1) + 0.5 * u + 0.25 * np.roll(u, -1, 1)
    assert np.abs(spectral_conv(on_grid(f, 256), w)[:, ::8] - spectral_conv(on_grid(f, 32), w)).max() < 1e-10
    assert np.abs(stencil(on_grid(f, 256))[:, ::8] - stencil(on_grid(f, 32))).max() > 0.1
    assert correct("fno-resolution") == [0]


def test_truncation_advection():
    train, rough = sample(200), sample(50, K=30, decay=1.0)
    k12, k30 = np.arange(1, 13), np.arange(1, 31)
    adv = lambda c, k: c * np.exp(-2j * np.pi * k * 0.3)
    heat = lambda c, k: c * np.exp(-(2 * np.pi * k) ** 2 * 0.002)
    for op, big in [(adv, True), (heat, False)]:
        w = fit(on_grid(train, 64), on_grid(op(train, k12), 64), 12)
        e64 = rel(spectral_conv(on_grid(rough, 64), w), on_grid(op(rough, k30), 64))
        e256 = rel(spectral_conv(on_grid(rough, 256), w), on_grid(op(rough, k30), 256))
        assert abs(e64 - e256) < 1e-6                          # 해상도 불변
        assert (e64 > 0.05) if big else (e64 < 1e-5)
    assert correct("truncation-advection") == [1]


def test_rollout_growth():
    q = Q["rollout-growth"]
    assert abs(1.0037**1000 - q["answer"]) <= q["tolerance"]


def test_unseen_modes_are_noise():
    train = sample(200)
    u_in = on_grid(train, 64)
    a = np.fft.rfft(u_in, axis=1)[:, 13:17] / 64
    assert np.abs(a).max() < 1e-12                            # 입력에 파수 13–16 성분이 없다
    assert correct("unseen-modes") == [0]


def test_other_answers():
    assert correct("operator-problem") == [0]
    assert correct("deeponet-sensors") == [0]
