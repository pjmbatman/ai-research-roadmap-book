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


mod = load()


def seqs_example(seed=0, lengths=(3, 7, 5), C=2):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n, C)) for n in lengths]


def test_shapes():
    x, mask, lengths = mod.pad_batch(seqs_example())
    assert np.shape(x) == (3, 7, 2), f"x의 shape은 (B, T_max, C) = (3, 7, 2)여야 합니다. 현재 {np.shape(x)}"
    assert np.shape(mask) == (3, 7) and np.asarray(mask).dtype == bool, "mask는 (B, T_max) bool 배열이어야 합니다"
    assert list(lengths) == [3, 7, 5]


def test_values_and_padding():
    seqs = seqs_example()
    x, mask, _ = mod.pad_batch(seqs, pad_value=-9.0)
    for i, s in enumerate(seqs):
        n = len(s)
        assert np.allclose(x[i, :n], s), "앞쪽에는 원래 값이 그대로 있어야 합니다"
        assert np.all(x[i, n:] == -9.0), "뒤쪽은 pad_value로 채워야 합니다"
        assert mask[i, :n].all() and not mask[i, n:].any(), "mask는 실제 값 위치에서만 True입니다"


def test_inputs_not_modified():
    seqs = seqs_example()
    copies = [s.copy() for s in seqs]
    x, _, _ = mod.pad_batch(seqs)
    x[...] = 0
    assert all(np.array_equal(a, b) for a, b in zip(seqs, copies)), "입력 배열과 메모리를 공유하거나 바꾸면 안 됩니다"


def test_masked_mse_ignores_padding_values():
    seqs = seqs_example(1)
    target_a, mask, _ = mod.pad_batch(seqs, pad_value=0.0)
    target_b, _, _ = mod.pad_batch(seqs, pad_value=1e6)
    pred = np.zeros_like(target_a)
    a, b = mod.masked_mse(pred, target_a, mask), mod.masked_mse(pred, target_b, mask)
    assert np.isclose(a, b), "padding 값이 바뀌어도 손실은 같아야 합니다"
    want = np.mean(np.concatenate([s.ravel() for s in seqs]) ** 2)
    assert np.isclose(a, want), "실제 값 원소 전체에 대한 제곱오차 평균이어야 합니다"


def test_masked_mse_differs_from_naive():
    seqs = [np.full((2, 1), 3.0), np.full((6, 1), 3.0)]
    x, mask, _ = mod.pad_batch(seqs)
    pred = np.zeros_like(x)
    assert np.isclose(mod.masked_mse(pred, x, mask), 9.0)
    assert not np.isclose(np.mean((pred - x) ** 2), 9.0)   # padding을 섞으면 틀린 값


def test_all_false_mask_raises():
    x = np.zeros((2, 3, 1))
    with pytest.raises(ValueError):
        mod.masked_mse(x, x, np.zeros((2, 3), dtype=bool))
