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


def batch(T=12, pad_value=0.0, seed=0):
    rng = np.random.default_rng(seed)
    lengths = np.array([12, 5, 2, 8])
    mask = np.arange(T)[None, :] < lengths[:, None]           # (B, T)
    loss = rng.uniform(0.2, 2.0, size=(4, T))
    loss[~mask] = pad_value
    return loss, mask, lengths


def test_token_mean_value():
    loss, mask, lengths = batch()
    expected = sum(loss[i, :l].sum() for i, l in enumerate(lengths)) / lengths.sum()
    assert np.isclose(mod.masked_token_mean(loss, mask), expected)


def test_sequence_mean_value():
    loss, mask, lengths = batch()
    expected = np.mean([loss[i, :l].mean() for i, l in enumerate(lengths)])
    assert np.isclose(mod.masked_sequence_mean(loss, mask), expected)
    assert not np.isclose(expected, mod.masked_token_mean(loss, mask)), "두 집계 방식은 일반적으로 다른 값을 준다"


@pytest.mark.parametrize("pad_value", [0.0, 1e6, np.nan])
def test_padding_values_ignored(pad_value):
    loss0, mask, _ = batch()
    loss1, _, _ = batch(pad_value=pad_value)
    for f in (mod.masked_token_mean, mod.masked_sequence_mean):
        assert np.isclose(f(loss1, mask), f(loss0, mask)), "padding 위치의 값은 결과에 영향을 주면 안 됩니다 (nan 포함)"


def test_padding_length_invariance():
    loss, mask, lengths = batch(T=12)
    loss_long = np.pad(loss, ((0, 0), (0, 20)), constant_values=7.0)
    mask_long = np.pad(mask, ((0, 0), (0, 20)), constant_values=False)
    assert np.isclose(mod.masked_token_mean(loss_long, mask_long), mod.masked_token_mean(loss, mask))
    assert np.isclose(mod.masked_sequence_mean(loss_long, mask_long), mod.masked_sequence_mean(loss, mask))


def test_empty_sequence_skipped():
    loss, mask, _ = batch()
    mask = mask.copy()
    mask[2] = False                                            # 실제 token이 없는 시퀀스
    v = mod.masked_sequence_mean(loss, mask)
    assert np.isfinite(v), "token이 없는 시퀀스는 평균에서 빼야 합니다"


def test_microbatch_consistency():
    loss, mask, _ = batch(pad_value=np.nan)
    s1, c1 = mod.masked_token_sum_count(loss[:2], mask[:2])
    s2, c2 = mod.masked_token_sum_count(loss[2:], mask[2:])
    assert np.isclose((s1 + s2) / (c1 + c2), mod.masked_token_mean(loss, mask)), \
        "micro-batch의 합과 개수를 모아 나누면 전체 batch의 token 평균과 같아야 합니다"
    naive = 0.5 * (s1 / c1 + s2 / c2)                          # micro-batch 평균의 평균
    assert not np.isclose(naive, mod.masked_token_mean(loss, mask))
