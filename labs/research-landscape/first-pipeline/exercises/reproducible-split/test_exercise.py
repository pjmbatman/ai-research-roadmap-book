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


split_indices = load().split_indices


@pytest.mark.parametrize("n,fractions", [(300, (0.6, 0.2, 0.2)), (101, (0.7, 0.15, 0.15)), (10, (0.5, 0.5, 0.0))])
def test_sizes_disjoint_complete(n, fractions):
    tr, va, te = split_indices(n, fractions, seed=0)
    assert len(tr) == int(fractions[0] * n), f"학습 크기는 int(f_train * n) = {int(fractions[0] * n)}입니다. 현재 {len(tr)}"
    assert len(va) == int(fractions[1] * n), f"검증 크기는 int(f_val * n) = {int(fractions[1] * n)}입니다. 현재 {len(va)}"
    allidx = np.concatenate([tr, va, te])
    assert len(allidx) == n, "세 배열을 합치면 n개여야 합니다(나머지는 모두 테스트)"
    assert np.array_equal(np.sort(allidx), np.arange(n)), "0..n-1이 빠짐없이, 겹치지 않게 들어 있어야 합니다"


def test_same_seed_same_split():
    a = split_indices(300, (0.6, 0.2, 0.2), seed=7)
    b = split_indices(300, (0.6, 0.2, 0.2), seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "같은 seed이면 같은 분할이어야 합니다"


def test_different_seed_different_split():
    a = split_indices(300, (0.6, 0.2, 0.2), seed=1)
    b = split_indices(300, (0.6, 0.2, 0.2), seed=2)
    assert not np.array_equal(a[2], b[2]), "다른 seed이면 분할이 달라져야 합니다"


def test_shuffled_not_sorted():
    tr, _, _ = split_indices(300, (0.6, 0.2, 0.2), seed=0)
    assert not np.array_equal(tr, np.arange(180)), "섞지 않고 앞에서 자르면 데이터의 저장 순서가 분할을 정합니다"


def test_independent_of_global_state():
    np.random.seed(0)
    a = split_indices(300, (0.6, 0.2, 0.2), seed=3)
    np.random.seed(12345)
    np.random.rand(17)                                       # 다른 코드가 먼저 난수를 뽑았다
    b = split_indices(300, (0.6, 0.2, 0.2), seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "전역 난수 상태가 달라도 같은 seed이면 같은 분할이어야 합니다"


def test_does_not_touch_global_state():
    np.random.seed(42)
    expected = np.random.rand(3)
    np.random.seed(42)
    split_indices(300, (0.6, 0.2, 0.2), seed=0)
    assert np.allclose(np.random.rand(3), expected), "함수가 전역 난수 상태를 바꾸면 안 됩니다"
