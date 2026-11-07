import importlib.util
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


kahan_sum = load().kahan_sum


def naive_sum(a):
    s = np.float32(0)
    for v in a:
        s += v
    return s


def rel_err(s, a):
    exact = np.sum(a.astype(np.float64))
    return abs(float(s) - exact) / abs(exact)


def test_small_exact_case():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert float(kahan_sum(a)) == 6.0


def test_returns_float32():
    a = np.ones(10, dtype=np.float32)
    assert np.asarray(kahan_sum(a)).dtype == np.float32, "중간 계산과 반환값은 float32여야 합니다"


def test_much_better_than_naive():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, 20_000).astype(np.float32)       # (20000,)
    err_k, err_n = rel_err(kahan_sum(a), a), rel_err(naive_sum(a), a)
    assert err_k < 1e-7, f"Kahan 합의 상대 오차가 float32 eps 수준이어야 합니다. 현재 {err_k:.1e}"
    assert err_k < err_n / 10, f"순서대로 더한 합({err_n:.1e})보다 크게 정확해야 합니다. 현재 {err_k:.1e}"


def test_large_plus_many_small():
    a = np.concatenate([[1e4], np.full(10_000, 1e-4)]).astype(np.float32)   # (10001,)
    # 1e4 근처 float32의 ulp는 약 1e-3이라, 순서대로 더하면 1e-4가 모두 반올림으로 사라진다
    assert float(naive_sum(a)) == 1e4
    assert abs(float(kahan_sum(a)) - 10001.0) < 2e-3, "작은 항 1만 개의 합(약 1)이 보존되어야 합니다"
