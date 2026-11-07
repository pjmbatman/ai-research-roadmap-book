"""19.8 퀴즈(content/19-generative-and-foundation-models/08-self-supervised-learning/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/19-generative-and-foundation-models/08-self-supervised-learning/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def nt_xent(z1, z2, tau):
    z = np.vstack([z1, z2])
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    S = z @ z.T / tau
    np.fill_diagonal(S, -np.inf)
    N = len(z1)
    pos = np.r_[np.arange(N, 2 * N), np.arange(N)]
    m = S.max(1, keepdims=True)
    return np.mean(np.log(np.exp(S - m).sum(1)) + m[:, 0] - S[np.arange(2 * N), pos])


def eff_rank(z):
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    p = s / s.sum()
    return np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))


def test_collapse_loss_value():
    z = np.ones((128, 16))
    for tau in (0.1, 0.5, 1.0):
        assert np.isclose(nt_xent(z, z, tau), np.log(255))
    q = Q["collapse-loss-value"]
    assert abs(np.log(255) - q["answer"]) <= q["tolerance"]


def test_alignment_loss_prefers_collapse():
    base = rng.normal(size=(64, 8))
    healthy = np.mean((base + 0.2 * rng.normal(size=base.shape) - base - 0.2 * rng.normal(size=base.shape)) ** 2)
    collapsed = np.mean((np.ones((64, 8)) - np.ones((64, 8))) ** 2)
    assert collapsed < healthy
    assert correct("noncontrastive-collapse") == [1]


def test_effective_rank_blind_to_scale():
    collapsed = np.ones((256, 16)) + 1e-3 * rng.normal(size=(256, 16))
    assert eff_rank(collapsed) > 15                          # 크기를 보지 않는다
    assert collapsed.std(0).mean() < 0.01                    # 표준편차는 붕괴를 드러낸다
    assert correct("effective-rank-blind") == [1]


def test_shortcut_has_lower_loss():
    n = 256
    sem = rng.normal(size=(n, 4)); tag = rng.normal(size=(n, 4))
    v1, v2 = sem + 0.7 * rng.normal(size=sem.shape), sem + 0.7 * rng.normal(size=sem.shape)
    assert nt_xent(tag, tag, 0.1) < nt_xent(v1, v2, 0.1)     # 표지를 공유하면 shortcut의 손실이 더 낮다
    assert nt_xent(tag, rng.normal(size=tag.shape), 0.1) > nt_xent(v1, v2, 0.1)   # 표지를 무작위화하면 반대
    assert correct("shortcut-low-loss") == [1]


def test_text_answers():
    assert correct("augmentation-invariance") == [0, 1]
    assert correct("label-budget") == [1]
    assert correct("fair-pretraining-comparison") == [0, 1, 2]
