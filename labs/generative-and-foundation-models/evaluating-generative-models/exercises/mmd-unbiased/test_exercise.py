import importlib.util
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mmd2_unbiased = load().mmd2_unbiased


def naive(x, y, bw):                                  # 정의를 그대로 옮긴 이중 반복문
    k = lambda a, b: math.exp(-sum((ai - bi) ** 2 for ai, bi in zip(a, b)) / (2 * bw**2))
    m, n = len(x), len(y)
    sxx = sum(k(x[i], x[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    syy = sum(k(y[i], y[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    sxy = sum(k(x[i], y[j]) for i in range(m) for j in range(n)) / (m * n)
    return sxx + syy - 2 * sxy


def test_small_example():
    x, y = np.array([[0.0], [1.0], [2.0]]), np.array([[0.0], [3.0]])
    got = mmd2_unbiased(x, y, 1.0)
    assert np.isclose(got, naive(x.tolist(), y.tolist(), 1.0)), f"정의대로 계산한 값 {naive(x.tolist(), y.tolist(), 1.0):.6f}과 다릅니다: {got}"


def test_unequal_sizes_and_symmetry():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(7, 2)), rng.normal(1.0, 1.0, size=(4, 2))
    assert np.isclose(mmd2_unbiased(x, y, 0.8), naive(x.tolist(), y.tolist(), 0.8)), "m ≠ n일 때 값이 다릅니다"
    assert np.isclose(mmd2_unbiased(x, y, 0.8), mmd2_unbiased(y, x, 0.8)), "인자의 순서를 바꿔도 값이 같아야 합니다"


def test_same_distribution_near_zero_can_be_negative():
    rng = np.random.default_rng(1)
    vals = [mmd2_unbiased(rng.normal(size=(40, 2)), rng.normal(size=(40, 2)), 1.0) for _ in range(30)]
    assert abs(np.mean(vals)) < 0.01, "같은 분포에서는 평균이 0 근처여야 합니다(불편 추정량)"
    assert min(vals) < 0, "대각 항을 뺀 불편 추정량은 같은 분포에서 음수도 나옵니다. 대각 항을 빼지 않았는지 확인하세요"


def test_mean_shift_increases():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 2))
    small = mmd2_unbiased(x, rng.normal(0.5, 1.0, size=(100, 2)), 1.0)
    large = mmd2_unbiased(x, rng.normal(2.0, 1.0, size=(100, 2)), 1.0)
    assert small > 0.02 and large > small, f"평균 차이가 클수록 커져야 합니다: {small:.4f}, {large:.4f}"


def test_wide_bandwidth_blind():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(60, 2)), rng.normal(0.5, 1.0, size=(60, 2))
    assert abs(mmd2_unbiased(x, y, 100.0)) < 1e-3, "대역폭이 차이의 척도보다 훨씬 크면 값이 0에 가까워야 합니다"
