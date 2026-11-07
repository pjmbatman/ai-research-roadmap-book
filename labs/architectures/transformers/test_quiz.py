"""18.4 퀴즈(content/18-architectures/04-transformers/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/18-architectures/04-transformers/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def attn(X, W, mask=None):
    Wq, Wk, Wv = W
    s = (X @ Wq) @ (X @ Wk).T / np.sqrt(Wq.shape[1])
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    return softmax(s) @ (X @ Wv)


def test_why_scale_sqrt_d():
    D = 512
    q, k = rng.normal(size=(20000, D)), rng.normal(size=(20000, D))
    var = np.var(np.sum(q * k, axis=1))
    assert 0.9 * D < var < 1.1 * D                                   # 분산 ≈ D
    assert 0.9 < np.var(np.sum(q * k, axis=1) / np.sqrt(D)) < 1.1      # 나누면 ≈ 1
    s = rng.normal(size=(16, 16)) * 5
    assert np.allclose(softmax(s).sum(-1), 1) and np.allclose(softmax(s / 7).sum(-1), 1)   # 행 합은 스케일과 무관
    assert correct("why-scale-sqrt-d") == [0]


def test_self_attention_permutation():
    D, T = 8, 6
    W = [rng.normal(size=(D, D)) for _ in range(3)]
    X = rng.normal(size=(T, D))
    P = np.eye(T)[rng.permutation(T)]
    assert np.allclose(attn(P @ X, W), P @ attn(X, W))
    assert not np.allclose(attn(P @ X, W), attn(X, W))
    assert correct("self-attention-permutation") == [0]


def test_rope_relative():
    def rot(x, pos):
        th = pos * 10000.0 ** (-np.arange(0, len(x), 2) / len(x))
        out = np.empty_like(x)
        out[0::2] = x[0::2] * np.cos(th) - x[1::2] * np.sin(th)
        out[1::2] = x[0::2] * np.sin(th) + x[1::2] * np.cos(th)
        return out
    q, k = rng.normal(size=16), rng.normal(size=16)
    vals = [rot(q, m) @ rot(k, m - 4) for m in (4, 9, 30, 77)]
    assert np.allclose(vals, vals[0])
    pe = lambda p: np.sin(p * np.arange(1, 17))                    # 더하는 방식
    add = [(q + pe(m)) @ (k + pe(m - 4)) for m in (4, 9, 30)]
    assert not np.allclose(add, add[0])
    assert correct("rope-relative") == [0]


def test_padding_mask_missing():
    D, T = 8, 5
    W = [rng.normal(size=(D, D)) for _ in range(3)]
    X = rng.normal(size=(T, D))
    X[3:] = 0.0
    X2 = X.copy()
    X2[3:] = rng.normal(size=(2, D))
    mask = np.ones((T, T), bool)
    mask[:, 3:] = False
    assert not np.allclose(attn(X, W)[:3], attn(X2, W)[:3])          # mask 없으면 PAD 값이 섞임
    assert np.array_equal(attn(X, W, mask)[:3], attn(X2, W, mask)[:3])
    # PAD 임베딩이 0이어도 점수 0은 가중치 e^0 > 0을 받는다
    assert softmax(np.array([0.0, 1.0, 0.0]))[0] > 0
    assert correct("padding-mask-missing") == [0]


def test_score_matrix_memory():
    gb = 4096 ** 2 * 16 * 4 / 1e9
    q = Q["score-matrix-memory"]
    assert abs(gb - q["answer"]) <= q["tolerance"]


def test_kv_cache_cost():
    D, T = 8, 7
    W = [rng.normal(size=(D, D)) for _ in range(3)]
    X = rng.normal(size=(T, D))
    full = attn(X, W, np.tril(np.ones((T, T), bool)))
    Kc, Vc = [], []
    for t in range(T):
        Kc.append(X[t] @ W[1])
        Vc.append(X[t] @ W[2])
        s = (X[t] @ W[0]) @ np.array(Kc).T / np.sqrt(D)             # 점수 t+1개: O(t)
        assert s.shape == (t + 1,)
        assert np.allclose(softmax(s) @ np.array(Vc), full[t])
    assert correct("kv-cache-cost") == [0]


def test_attention_not_explanation():
    # 가중치가 큰 key라도 value가 작으면 출력 기여는 작다: 가중치 ≠ 기여
    w = softmax(np.array([3.0, 0.0, 0.0]))
    V = np.array([[0.01], [1.0], [1.0]])
    contrib = w[:, None] * V
    assert np.argmax(w) == 0 and np.argmax(np.abs(contrib)) != 0
    assert correct("attention-not-explanation") == [0, 1]
