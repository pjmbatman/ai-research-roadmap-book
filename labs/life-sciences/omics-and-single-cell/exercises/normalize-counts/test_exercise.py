import importlib.util
import os
import warnings
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


normalize_log1p = load().normalize_log1p


def test_hand_computed():
    C = np.array([[10, 1990], [0, 500]])
    out = normalize_log1p(C)
    assert np.shape(out) == (2, 2)
    assert np.isclose(out[0, 0], np.log(51)), "10 / 2000 × 1e4 = 50, log(1 + 50)"
    assert out[1, 0] == 0.0 and np.isclose(out[1, 1], np.log1p(1e4))


def test_depth_invariance():
    rng = np.random.default_rng(0)
    C = rng.poisson(3.0, size=(5, 40))
    assert np.allclose(normalize_log1p(C), normalize_log1p(2 * C)), "깊이만 두 배인 세포는 같은 값이어야 합니다"


def test_row_sums_before_log():
    rng = np.random.default_rng(1)
    C = rng.poisson(2.0, size=(6, 30)) + 1
    for ts in [1e4, 1e6]:
        out = normalize_log1p(C, target_sum=ts)
        assert np.allclose(np.expm1(out).sum(1), ts), "log 전의 세포별 총합은 target_sum이어야 합니다"


def test_empty_cell_no_nan():
    C = np.array([[0, 0, 0], [1, 2, 3]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # 0으로 나누기 경고도 실패로 본다
        out = normalize_log1p(C)
    assert np.all(np.isfinite(out)), "nan이나 inf가 있으면 안 됩니다"
    assert np.all(out[0] == 0), "총합이 0인 세포의 행은 0입니다"


def test_input_not_modified():
    C = np.array([[1, 2], [3, 4]])
    before = C.copy()
    normalize_log1p(C)
    assert np.array_equal(C, before), "입력 배열을 바꾸면 안 됩니다"
