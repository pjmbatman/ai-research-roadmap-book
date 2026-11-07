import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


attention = load().attention


def loop_attention(Q, K, V, mask=None):
    """2차원 Q (T, D), K (S, D), V (S, Dv)에 대해 정의를 반복문으로"""
    T, D = Q.shape
    S = K.shape[0]
    out, W = np.zeros((T, V.shape[1])), np.zeros((T, S))
    for t in range(T):
        s = [Q[t] @ K[j] / math.sqrt(D) if mask is None or mask[t, j] else -math.inf for j in range(S)]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        z = sum(e)
        W[t] = [v / z for v in e]
        out[t] = W[t] @ V
    return out, W


@pytest.mark.parametrize("T,S,D,Dv", [(1, 1, 2, 1), (3, 5, 4, 2), (6, 6, 8, 8)])
def test_matches_loop(T, S, D, Dv):
    rng = np.random.default_rng(T * 100 + S)
    Q, K, V = rng.normal(size=(T, D)), rng.normal(size=(S, D)), rng.normal(size=(S, Dv))
    out, w = attention(Q, K, V)
    ref_out, ref_w = loop_attention(Q, K, V)
    assert np.shape(out) == (T, Dv) and np.shape(w) == (T, S), "출력 (T, Dv), 가중치 (T, S)여야 합니다"
    assert np.allclose(w, ref_w), "가중치가 softmax(QKᵀ/√D)와 다릅니다. √D로 나눴는지, 마지막 축으로 softmax했는지 확인하세요"
    assert np.allclose(out, ref_out)


def test_batch_and_head_axes():
    rng = np.random.default_rng(1)
    Q, K, V = rng.normal(size=(2, 3, 4, 8)), rng.normal(size=(2, 3, 5, 8)), rng.normal(size=(2, 3, 5, 6))
    out, w = attention(Q, K, V)
    assert np.shape(out) == (2, 3, 4, 6) and np.shape(w) == (2, 3, 4, 5)
    for b in range(2):
        for h in range(3):
            ref_out, _ = loop_attention(Q[b, h], K[b, h], V[b, h])
            assert np.allclose(out[b, h], ref_out), "batch·head 항목끼리 섞이면 안 됩니다"


def test_rows_sum_to_one():
    rng = np.random.default_rng(2)
    _, w = attention(rng.normal(size=(5, 4)), rng.normal(size=(7, 4)), rng.normal(size=(7, 3)))
    assert np.allclose(np.sum(w, axis=-1), 1.0), "가중치의 각 행(query마다)의 합은 1이어야 합니다"


def test_mask_gives_exact_zero():
    rng = np.random.default_rng(3)
    T = 5
    mask = np.tril(np.ones((T, T), dtype=bool))
    Q, K, V = rng.normal(size=(T, 4)), rng.normal(size=(T, 4)), rng.normal(size=(T, 2))
    out, w = attention(Q, K, V, mask)
    assert np.all(w[~mask] == 0.0), "mask가 False인 칸의 가중치는 정확히 0이어야 합니다"
    assert np.allclose(out, loop_attention(Q, K, V, mask)[0])


def test_large_scores_are_stable():
    Q = np.array([[300.0, 0.0], [0.0, -300.0]])
    K = np.array([[300.0, 0.0], [-300.0, 0.0], [0.0, 300.0]])
    out, w = attention(Q, K, np.eye(3))
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(out)), "점수가 커도 nan/inf가 나오면 안 됩니다(최댓값을 빼세요)"
    assert np.allclose(w[0], [1.0, 0.0, 0.0])
