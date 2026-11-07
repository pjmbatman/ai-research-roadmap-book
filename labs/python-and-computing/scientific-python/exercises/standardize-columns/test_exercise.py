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


standardize = load().standardize


def test_column_statistics():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[5.0, -3.0, 100.0], scale=[1.0, 0.1, 20.0], size=(200, 3))   # (N, D)
    Z, mu, sigma = standardize(X)
    assert np.shape(Z) == (200, 3) and np.shape(mu) == (3,) and np.shape(sigma) == (3,)
    assert np.allclose(Z.mean(axis=0), 0), "열 평균이 0이 아닙니다. axis를 확인하세요"
    assert np.allclose(Z.std(axis=0), 1), "열 표준편차가 1이 아닙니다"
    assert np.allclose(Z * sigma + mu, X), "Z * sigma + mu로 원래 값이 복원되어야 합니다"


def test_square_matrix_uses_columns():
    # N == D이면 axis를 잘못 골라도 shape 오류가 나지 않는다. 값으로 확인한다
    X = np.array([[1.0, 10.0], [3.0, 30.0]])
    Z, mu, sigma = standardize(X)
    assert np.allclose(mu, [2.0, 20.0]), f"mu는 열 평균이어야 합니다. 현재 {mu}"
    assert np.allclose(sigma, [1.0, 10.0])
    assert np.allclose(Z, [[-1.0, -1.0], [1.0, 1.0]])


def test_input_unchanged():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 4))
    X0 = X.copy()
    standardize(X)
    assert np.array_equal(X, X0), "입력 배열이 바뀌었습니다. 제자리 연산(-=, /=)을 쓰지 않았는지 확인하세요"


def test_integer_input():
    X = np.array([[1, 2], [2, 4], [3, 6]], dtype=np.int64)
    X0 = X.copy()
    Z, mu, sigma = standardize(X)
    assert np.issubdtype(np.asarray(Z).dtype, np.floating), "결과는 실수 배열이어야 합니다"
    assert np.allclose(Z[:, 0], [-1.224744871391589, 0.0, 1.224744871391589])
    assert np.array_equal(X, X0)


def test_constant_column():
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    Z, mu, sigma = standardize(X)
    assert np.all(np.isfinite(Z)), "상수 열에서 0으로 나누면 nan이 생깁니다"
    assert np.allclose(Z[:, 1], 0) and np.isclose(sigma[1], 1.0)
