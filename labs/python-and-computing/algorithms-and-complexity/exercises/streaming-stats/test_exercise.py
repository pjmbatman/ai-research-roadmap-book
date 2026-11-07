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


streaming_mean_var = load().streaming_mean_var


def as_generator(x, n_chunks):
    for c in np.array_split(x, n_chunks):
        yield c


@pytest.mark.parametrize("n_chunks", [1, 3, 17])
def test_matches_full_computation(n_chunks):
    rng = np.random.default_rng(n_chunks)
    x = rng.normal(loc=2.0, scale=3.0, size=10_000)
    mean, var = streaming_mean_var(as_generator(x, n_chunks))
    assert np.isclose(mean, x.mean()) and np.isclose(var, x.var())


def test_unequal_chunks_and_empty_chunk():
    x = np.arange(10.0)
    chunks = [x[:1], np.array([]), x[1:7], x[7:]]
    mean, var = streaming_mean_var(iter(chunks))
    assert np.isclose(mean, 4.5) and np.isclose(var, 8.25), f"현재 ({mean}, {var})"


def test_single_pass():
    class OnePass:
        def __init__(self, parts):
            self.parts, self.used = parts, False

        def __iter__(self):
            assert not self.used, "입력을 두 번 순회했습니다"
            self.used = True
            return iter(self.parts)

    x = np.linspace(0, 1, 101)
    mean, var = streaming_mean_var(OnePass(np.array_split(x, 5)))
    assert np.isclose(mean, x.mean()) and np.isclose(var, x.var())


def test_numerically_stable_with_large_offset():
    rng = np.random.default_rng(0)
    x = 1e8 + rng.normal(size=200_000)          # 평균 1e8, 분산 약 1
    mean, var = streaming_mean_var(as_generator(x, 50))
    assert np.isclose(mean, x.mean(), rtol=0, atol=1e-6)
    assert abs(var - x.var()) < 1e-3, f"분산 {var} (정답 {x.var():.6f}). E[x^2] - E[x]^2 누적은 소거 오차로 무너집니다"


def test_empty_raises():
    with pytest.raises(ValueError):
        streaming_mean_var(iter([np.array([]), np.array([])]))
