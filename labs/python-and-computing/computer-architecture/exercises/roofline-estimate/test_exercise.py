import importlib.util
import math
import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


roofline = load().roofline

PEAK = 20e12      # 가상의 장치: 20 TFLOP/s
BW = 1e12         # 1 TB/s  → 경계(ridge) 연산 강도 20 FLOP/byte


def test_vector_add_is_memory_bound():
    n = 10**8
    t, intensity, bound = roofline(n, 3 * 4 * n, PEAK, BW)        # float32 z = x + y
    assert math.isclose(intensity, 1 / 12)
    assert bound == "memory"
    assert math.isclose(t, 1.2e-3), f"메모리 시간 1.2e9 byte / 1e12 byte/s = 1.2 ms. 현재 {t}"


def test_large_matmul_is_compute_bound():
    m = 8192
    flops, nbytes = 2 * m**3, 3 * 4 * m * m                       # float32, 세 행렬 한 번씩
    t, intensity, bound = roofline(flops, nbytes, PEAK, BW)
    assert bound == "compute"
    assert math.isclose(intensity, 2 * m / 12)
    assert math.isclose(t, flops / PEAK)


def test_small_matmul_is_memory_bound():
    m = 16
    t, intensity, bound = roofline(2 * m**3, 3 * 4 * m * m, PEAK, BW)
    assert intensity < PEAK / BW and bound == "memory", "작은 행렬 곱은 연산 강도가 경계보다 낮습니다"


def test_time_is_max_not_sum():
    t, _, _ = roofline(1e12, 1e12, 1e12, 1e12)                    # 두 시간이 모두 1초
    assert math.isclose(t, 1.0), "roofline은 두 시간이 겹친다고 보므로 합이 아니라 최댓값입니다"


def test_tie_is_compute():
    _, _, bound = roofline(2e12, 1e12, 2e12, 1e12)
    assert bound == "compute"


def test_zero_bytes():
    t, intensity, bound = roofline(1e9, 0, PEAK, BW)
    assert intensity == float("inf") and bound == "compute" and math.isclose(t, 1e9 / PEAK)


@pytest.mark.parametrize(
    "args",
    [(-1, 10, PEAK, BW), (10, -1, PEAK, BW), (0, 0, PEAK, BW), (10, 10, 0, BW), (10, 10, PEAK, -1)],
)
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        roofline(*args)
