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


def elem_strides(a):
    return tuple(s // a.itemsize for s in a.strides)


@pytest.mark.parametrize("shape", [(5,), (2, 3), (2, 3, 4), (4, 1, 3), ()])
def test_row_major_strides(shape):
    a = np.zeros(shape)
    assert tuple(mod.row_major_strides(shape)) == elem_strides(a), f"{shape}의 행 우선 stride가 다릅니다"


def test_row_major_example():
    assert tuple(mod.row_major_strides((2, 3, 4))) == (12, 4, 1)


@pytest.mark.parametrize("shape,dims", [((2, 3), (1, 0)), ((2, 3, 4), (0, 2, 1)), ((2, 3, 4), (2, 0, 1)), ((8, 5, 6, 3), (0, 3, 1, 2))])
def test_permute_strides(shape, dims):
    a = np.zeros(shape)
    p = a.transpose(dims)
    got_shape, got_strides = mod.permute_strides(shape, elem_strides(a), dims)
    assert tuple(got_shape) == p.shape, "permute 뒤 shape이 다릅니다"
    assert tuple(got_strides) == elem_strides(p), "permute 뒤 stride가 다릅니다"


@pytest.mark.parametrize("make", [
    lambda: np.zeros((2, 3, 4)),
    lambda: np.zeros((2, 3, 4)).transpose(1, 0, 2),
    lambda: np.zeros((3, 4)).T,
    lambda: np.zeros((4, 1, 3)),
    lambda: np.zeros((3, 1)).T,                   # (1, 3), 길이 1인 축 stride는 무시
    lambda: np.zeros((6, 4))[:, :2],              # 열 일부만: 행 사이에 빈틈
    lambda: np.zeros((6, 4))[::2],                # 행 건너뛰기
    lambda: np.zeros((6, 4))[1:3],                # 연속한 행 일부: contiguous
])
def test_is_contiguous_matches_numpy(make):
    a = make()
    want = a.flags["C_CONTIGUOUS"]
    got = mod.is_contiguous(a.shape, elem_strides(a))
    assert bool(got) == want, f"shape {a.shape}, stride {elem_strides(a)}: contiguous는 {want}여야 합니다"


def test_transpose_breaks_contiguity():
    shape = (2, 3)
    s = mod.row_major_strides(shape)
    assert mod.is_contiguous(shape, s)
    ps, pst = mod.permute_strides(shape, s, (1, 0))
    assert not mod.is_contiguous(ps, pst), "전치한 (3, 2) 텐서는 contiguous가 아닙니다"
