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


chunked_mean_var = load().chunked_mean_var


def gen(x, size):
    """x를 size개씩 잘라 한 번만 내놓는 generator."""
    for s in range(0, len(x), size):
        yield x[s:s + size]


@pytest.mark.parametrize("size", [1, 7, 1000, 5000])
def test_matches_numpy(size):
    x = np.random.default_rng(size).normal(3.0, 2.0, size=5000)
    n, mean, var = chunked_mean_var(gen(x, size))
    assert n == 5000, "전체 개수가 다릅니다"
    assert np.isclose(mean, x.mean()), "평균이 다릅니다"
    assert np.isclose(var, x.var()), "모분산(ddof=0)이 다릅니다"


def test_large_offset():
    x = 1e9 + np.random.default_rng(0).normal(0.0, 1.0, size=20_000)
    _, mean, var = chunked_mean_var(gen(x, 3000))
    assert np.isclose(mean, x.mean(), rtol=0, atol=1e-6)
    assert np.isclose(var, x.var(), rtol=1e-6), (
        f"offset 1e9 위에서 분산이 {var:.6g}로 나왔습니다(정답 {x.var():.6g}). E[x²] - E[x]²은 소거 오차가 큽니다")


def test_uneven_and_empty_chunks():
    x = np.random.default_rng(1).uniform(size=100)
    parts = [x[:3], x[3:3], x[3:90], np.array([]), x[90:]]
    n, mean, var = chunked_mean_var(iter(parts))
    assert n == 100 and np.isclose(mean, x.mean()) and np.isclose(var, x.var()), "빈 덩어리는 건너뛰어야 합니다"
