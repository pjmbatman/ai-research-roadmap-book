"""26.5 퀴즈(content/26-physics-and-engineering/05-physics-benchmarks/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/26-physics-and-engineering/05-physics-benchmarks/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def ftcs(u, r, steps):
    for _ in range(steps):
        u = u + r * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
    return u


def test_explicit_stability_ood():
    c = 3.4
    nu_max = Q["explicit-stability-ood"]["answer"]
    assert abs(nu_max - 0.5 / c) <= Q["explicit-stability-ood"]["tolerance"]
    u0 = np.random.default_rng(0).normal(size=64)
    assert np.abs(ftcs(u0, c * 0.1, 200)).max() <= np.abs(u0).max()          # 학습 범위: 안정
    assert np.abs(ftcs(u0, c * 0.3, 200)).max() > 1e3                         # nu = 0.3: 발산


def test_high_wavenumber_energy():
    N = 64
    x = np.arange(N) * 2 * np.pi / N
    ref = np.sin(x) + 0.05 * np.sin(10 * x)
    noisy = ref + 0.3 * np.sin(20 * x)                                        # 고파수 진동 추가
    smooth = np.sin(x) + 0.02 * np.sin(10 * x)                                # 고파수를 뭉갬

    def hi(u):
        E = np.abs(np.fft.rfft(u)) ** 2
        return E[8:].sum() / E[1:].sum()

    assert hi(noisy) / hi(ref) > 1 and hi(smooth) / hi(ref) < 1
    # L2 오차로는 고파수 차이가 작게 보인다
    assert np.linalg.norm(smooth - ref) / np.linalg.norm(ref) < 0.05
    assert correct("high-wavenumber-energy") == [0]


def test_choice_keys():
    assert correct("one-step-not-enough") == [0]
    assert correct("trajectory-split") == [0]
    assert correct("strong-baseline") == [0]
    assert correct("ood-axes") == [0, 1, 2]
