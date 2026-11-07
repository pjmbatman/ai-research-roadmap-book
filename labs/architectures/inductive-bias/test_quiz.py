"""18.1 퀴즈(content/18-architectures/01-inductive-bias/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/18-architectures/01-inductive-bias/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_locality_without_sharing():
    T, k = 100, 7
    mask = np.zeros((T, T), dtype=bool)                      # 출력 t가 보는 입력 위치
    for t in range(T):
        for j in range(-(k // 2), k // 2 + 1):
            if 0 <= t + j < T:                               # zero padding 칸은 입력이 아니지만 가중치는 존재
                mask[t, t + j] = True
    n_weights = T * k                                        # 위치마다 kernel 7개 (padding 칸에 곱해지는 것 포함)
    assert mask.sum() <= n_weights
    assert n_weights == Q["locality-without-sharing"]["answer"] == 700


def circ_conv(x, w):                                         # (C_in, T), (C_out, C_in, k) → (C_out, T)
    k = w.shape[-1]
    win = np.stack([np.roll(x, -(j - k // 2), axis=-1) for j in range(k)], -1)   # (C_in, T, k)
    return np.einsum("ctk,ock->ot", win, w)


def test_pooling_makes_invariant():
    x = rng.normal(size=(1, 16))
    w1, w2 = rng.normal(size=(4, 1, 3)), rng.normal(size=(5, 4, 3))
    f = lambda x: np.maximum(0, circ_conv(np.maximum(0, circ_conv(x, w1)), w2)).mean(axis=-1)
    feat = lambda x: np.maximum(0, circ_conv(np.maximum(0, circ_conv(x, w1)), w2))
    xs = np.roll(x, 3, axis=-1)
    assert np.allclose(feat(xs), np.roll(feat(x), 3, axis=-1))      # pooling 전: 등변
    assert np.allclose(f(xs), f(x))                                  # pooling 후: 불변
    assert correct("pooling-makes-invariant") == [0]


def test_augmentation_vs_constraint():
    assert correct("augmentation-vs-constraint") == [0]


def test_mismatched_bias_plateau():
    T, K = 32, 3
    A, b = rng.normal(size=(K, 100)), rng.normal(size=100)
    feat = lambda X: np.maximum(0, np.stack([np.roll(X, -j, axis=1) for j in range(K)], -1) @ A + b).sum(1)
    w_pos = np.linspace(0, 2, T)
    target = lambda X: np.maximum(0, X * np.roll(X, -1, axis=1)) @ w_pos / np.sqrt(T)
    Xte = rng.normal(size=(2000, T))
    def r2(N):
        X = rng.normal(size=(N, T))
        F, G = np.c_[feat(X), np.ones(N)], np.c_[feat(Xte), np.ones(len(Xte))]
        w = np.linalg.solve(F.T @ F + np.eye(F.shape[1]), F.T @ target(X))
        return 1 - np.mean((G @ w - target(Xte)) ** 2) / np.var(target(Xte))
    r_mid, r_big = r2(2000), r2(8000)
    assert r_big - r_mid < 0.05 and r_big < 0.8                      # 멈춤(plateau)
    # 이동 불변 모델은 원형 이동한 입력에 같은 예측을 내지만 목표는 달라진다
    X = rng.normal(size=(5, T))
    assert np.allclose(feat(X), feat(np.roll(X, 5, axis=1)))
    assert not np.allclose(target(X), target(np.roll(X, 5, axis=1)))
    assert correct("mismatched-bias-plateau") == [0]


def test_match_data_to_bias():
    assert correct("match-data-to-bias") == [0, 1]
