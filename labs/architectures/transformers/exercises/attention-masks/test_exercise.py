import importlib.util
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


make_mask = load().make_mask
LENGTHS, T = np.array([5, 3, 1]), 5


def attention(X, mask):                                   # X (B, T, D), mask (B, T, T)
    rng = np.random.default_rng(0)
    D = X.shape[-1]
    Wq, Wk, Wv = (rng.normal(size=(D, D)) for _ in range(3))
    s = (X @ Wq) @ np.swapaxes(X @ Wk, -1, -2) / np.sqrt(D)
    s = np.where(mask, s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True))
    return (w / w.sum(-1, keepdims=True)) @ (X @ Wv)


@pytest.mark.parametrize("causal", [False, True])
def test_shape_dtype(causal):
    m = make_mask(LENGTHS, T, causal)
    assert np.shape(m) == (3, T, T), f"shape은 (B, T, T) = (3, {T}, {T})여야 합니다"
    assert np.asarray(m).dtype == bool, "bool 배열이어야 합니다"


def test_padding_only():
    m = make_mask(LENGTHS, T, causal=False)
    for b, L in enumerate(LENGTHS):
        assert m[b][:, :L].all(), "PAD가 아닌 key는 모든 query가 볼 수 있어야 합니다"
        assert not m[b][:, L:].any(), "PAD key는 어떤 query도 볼 수 없어야 합니다"


def test_causal_and_padding():
    m = make_mask(LENGTHS, T, causal=True)
    for b, L in enumerate(LENGTHS):
        expect = np.tril(np.ones((T, T), dtype=bool))
        expect[:, L:] = False
        assert np.array_equal(m[b], expect), f"시퀀스 {b}(길이 {L})의 mask가 다릅니다"


@pytest.mark.parametrize("causal", [False, True])
def test_no_empty_rows(causal):
    m = make_mask(LENGTHS, T, causal)
    assert m.any(axis=-1).all(), "모든 query 행에 볼 수 있는 key가 하나 이상 있어야 합니다(없으면 softmax가 nan)"


def test_pad_values_do_not_leak():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, T, 4))
    X2 = X.copy()
    for b, L in enumerate(LENGTHS):
        X2[b, L:] = rng.normal(size=(T - L, 4)) * 10
    for causal in (False, True):
        m = make_mask(LENGTHS, T, causal)
        y, y2 = attention(X, m), attention(X2, m)
        for b, L in enumerate(LENGTHS):
            assert np.allclose(y[b, :L], y2[b, :L]), "PAD 값이 실제 토큰의 출력에 섞이면 안 됩니다"


def test_future_does_not_leak():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3, T, 4))
    X2 = X.copy()
    X2[:, 3] += 5.0
    m = make_mask(np.array([T, T, T]), T, causal=True)
    assert np.allclose(attention(X, m)[:, :3], attention(X2, m)[:, :3]), "causal mask에서는 미래 토큰이 과거 출력에 영향을 주면 안 됩니다"
