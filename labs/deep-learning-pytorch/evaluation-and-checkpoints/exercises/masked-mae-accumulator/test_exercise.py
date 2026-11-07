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


masked_mae = load().masked_mae


def make_batch(rng, B, T):
    pred, target = rng.normal(size=(B, T)), rng.normal(size=(B, T))
    lengths = rng.integers(1, T + 1, size=B)
    mask = (np.arange(T) < lengths[:, None]).astype(float)       # (B, T)
    return pred, target, mask


def reference(batches):
    errs = np.concatenate([np.abs(p - t)[m > 0] for p, t, m in batches])
    return errs.mean()


def test_single_full_batch():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.zeros((2, 2))
    assert masked_mae([(pred, target, None)]) == pytest.approx(2.5)


def test_uneven_batches_match_concatenation():
    rng = np.random.default_rng(0)
    batches = [make_batch(rng, B, T) for B, T in [(8, 5), (8, 7), (3, 4)]]
    assert masked_mae(batches) == pytest.approx(reference(batches)), \
        "batch마다 평균을 내고 다시 평균하면 batch 크기·관측 수가 다를 때 값이 달라집니다"


def test_not_mean_of_batch_means():
    big = (np.ones((9, 1)), np.zeros((9, 1)), None)               # 오차 1이 9개
    small = (np.full((1, 1), 11.0), np.zeros((1, 1)), None)       # 오차 11이 1개
    assert masked_mae([big, small]) == pytest.approx(2.0), "정답은 (9×1 + 11) / 10 = 2.0입니다. 6.0이면 batch 평균의 평균입니다"


def test_padding_values_are_ignored():
    rng = np.random.default_rng(1)
    pred, target, mask = make_batch(rng, 6, 5)
    expected = masked_mae([(pred, target, mask)])
    pred2 = pred.copy()
    pred2[mask == 0] = np.nan                                     # padding 위치에 NaN
    assert masked_mae([(pred2, target, mask)]) == pytest.approx(expected), "padding 위치의 NaN이 결과를 오염시켰습니다"
    pred3 = pred.copy()
    pred3[mask == 0] = 1e6
    assert masked_mae([(pred3, target, mask)]) == pytest.approx(expected), "padding 위치의 값이 결과에 들어갔습니다"


def test_empty_raises():
    with pytest.raises(ValueError):
        masked_mae([(np.ones((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))])
