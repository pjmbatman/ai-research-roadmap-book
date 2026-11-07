"""7.7 퀴즈(content/07-linear-algebra/07-tensors-and-einsum/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/07-linear-algebra/07-tensors-and-einsum/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_einsum_frobenius():
    A, C = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    v = np.einsum("ij,ij->", A, C)
    assert np.isclose(v, np.trace(A.T @ C))                    # 선택지 0
    assert np.ndim(v) == 0                                     # 선택지 1(원소별 곱 행렬)이 아님
    S, U = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    assert not np.isclose(np.einsum("ij,ij->", S, U), np.trace(S @ U))   # 선택지 3
    assert correct("einsum-frobenius") == [0]


def test_reshape_vs_transpose():
    B, T, D = 3, 3, 2                                          # B = T여도 다르다
    X = rng.normal(size=(B, T, D))
    good = X.transpose(1, 0, 2)
    assert np.allclose(good[1, 2], X[2, 1])
    assert not np.allclose(X.reshape(T, B, D), good)
    assert correct("reshape-vs-transpose") == [0]


def test_broadcast_result_shape():
    a, b = np.zeros((5, 1, 3)), np.zeros((4, 3))
    assert (a - b).shape == (5, 4, 3)
    assert correct("broadcast-result-shape") == [0]


def test_einsum_bilinear_spec():
    x, W, z = rng.normal(size=(4, 2)), rng.normal(size=(3, 2, 5)), rng.normal(size=(4, 5))
    ref = np.array([[x[b] @ W[k] @ z[b] for k in range(3)] for b in range(4)])
    assert np.allclose(np.einsum("bi,kij,bj->bk", x, W, z), ref)
    assert np.einsum("bi,kij,bj->bkij", x, W, z).shape == (4, 3, 2, 5)
    assert np.einsum("bi,kij,bj->k", x, W, z).shape == (3,)
    with pytest.raises(ValueError):                            # I ≠ J이면 kji는 오류
        np.einsum("bi,kji,bj->bk", x, W, z)
    Wsq, zsq = rng.normal(size=(3, 2, 2)), rng.normal(size=(4, 2))
    ref_sq = np.einsum("bi,kij,bj->bk", x, Wsq, zsq)
    assert not np.allclose(np.einsum("bi,kji,bj->bk", x, Wsq, zsq), ref_sq)   # I = J이면 조용히 틀림
    assert correct("einsum-bilinear-spec") == [0]


def test_contraction_order():
    n = 1000
    cost_ab_first = n * n * n + n * n
    cost_bx_first = 2 * n * n
    assert cost_bx_first == 2_000_000
    assert round(cost_ab_first / cost_bx_first) == 500
    path, _ = np.einsum_path("ij,jk,k->i", np.zeros((n, n)), np.zeros((n, n)), np.zeros(n), optimize="optimal")
    assert path[1] == (1, 2)                                   # 최적 경로는 B와 x를 먼저 축약
    assert correct("contraction-order") == [0]


def test_cp_parameter_count():
    I, J, K, R = 10, 20, 30, 5
    a, b, c = rng.normal(size=(I, R)), rng.normal(size=(J, R)), rng.normal(size=(K, R))
    T = np.einsum("ir,jr,kr->ijk", a, b, c)
    assert T.shape == (I, J, K)
    assert a.size + b.size + c.size == Q["cp-parameter-count"]["answer"] == 300
