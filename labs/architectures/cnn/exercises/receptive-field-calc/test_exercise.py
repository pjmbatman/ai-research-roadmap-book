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


receptive_field = load().receptive_field


def conv1d(x, k, s, d):
    """kernel이 모두 1인 valid 합성곱"""
    span = d * (k - 1) + 1
    return np.array([x[t:t + span:d].sum() for t in range(0, len(x) - span + 1, s)])


def measured(layers, n=120):
    """입력 칸 하나씩 바꿔 보며 출력 첫 칸·둘째 칸이 영향을 받는 입력 범위를 직접 센다"""
    def net(x):
        for k, s, d in layers:
            x = conv1d(x, k, s, d)
        return x
    base = net(np.zeros(n))
    assert len(base) >= 2
    hit = [[i for i in range(n) if net(np.eye(n)[i])[j] != base[j]] for j in (0, 1)]
    return hit[0][-1] - hit[0][0] + 1, hit[1][0] - hit[0][0]


def test_empty():
    assert tuple(receptive_field([])) == (1, 1)


def test_plain_stack():
    assert tuple(receptive_field([(3, 1, 1)] * 5)) == (11, 1)


def test_tcn_dilations():
    r, J = receptive_field([(2, 1, 2 ** i) for i in range(4)])
    assert (r, J) == (16, 1), f"kernel 2, dilation 1·2·4·8이면 r = 16이어야 합니다. 현재 {r}"


def test_stride_multiplies_later_layers():
    r, J = receptive_field([(3, 2, 1), (3, 2, 1), (3, 1, 1)])
    assert J == 4, f"jump는 stride의 곱 2·2·1 = 4여야 합니다. 현재 {J}"
    assert r == 15, f"r = 1 + 2·1 + 2·2 + 2·4 = 15여야 합니다. 현재 {r}"


@pytest.mark.parametrize("seed", range(6))
def test_matches_measurement(seed):
    rng = np.random.default_rng(seed)
    layers = [(int(rng.integers(1, 5)), int(rng.integers(1, 3)), int(rng.integers(1, 3))) for _ in range(4)]
    assert tuple(receptive_field(layers)) == measured(layers), f"층 구성 {layers}에서 직접 센 값과 다릅니다"
