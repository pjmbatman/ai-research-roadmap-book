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


latin_hypercube = load().latin_hypercube


@pytest.mark.parametrize("n,d", [(5, 1), (16, 2), (50, 6)])
def test_one_point_per_stratum(n, d):
    S = np.asarray(latin_hypercube(n, d, seed=0))
    assert S.shape == (n, d), f"shape은 ({n}, {d})여야 합니다. 현재 {S.shape}"
    assert np.all((S >= 0) & (S < 1)), "모든 값은 [0, 1) 안에 있어야 합니다"
    for j in range(d):
        bins = np.sort(np.floor(S[:, j] * n).astype(int))
        assert np.array_equal(bins, np.arange(n)), f"{j}번째 축에서 구간마다 정확히 한 점이 있어야 합니다"


def test_reproducible_with_seed():
    a, b = latin_hypercube(10, 3, seed=42), latin_hypercube(10, 3, seed=42)
    assert np.array_equal(a, b), "같은 seed는 같은 표본을 내야 합니다"
    assert not np.array_equal(a, latin_hypercube(10, 3, seed=43)), "다른 seed는 다른 표본을 내야 합니다"


def test_axes_are_paired_randomly():
    S = latin_hypercube(30, 2, seed=1)
    order0, order1 = np.argsort(S[:, 0]), np.argsort(S[:, 1])
    assert not np.array_equal(order0, order1), "축마다 순열을 따로 만들어야 합니다(대각선 위의 점들이 되면 안 됩니다)"


def test_position_inside_stratum_is_random():
    S = latin_hypercube(20, 2, seed=3)
    frac = S * 20 - np.floor(S * 20)                        # 구간 안의 상대 위치
    assert frac.std() > 0.1, "점이 구간의 같은 위치(예: 중앙)에만 놓이면 안 됩니다"
