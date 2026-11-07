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


standardize_split = load().standardize_split
rng = np.random.default_rng(0)


def test_train_is_standardized():
    X_tr = rng.normal([5.0, -2.0], [3.0, 0.5], size=(200, 2))
    Z_tr, _, mean, std = standardize_split(X_tr, rng.normal(size=(10, 2)))
    assert np.allclose(Z_tr.mean(axis=0), 0.0), "표준화한 train의 열 평균은 0이어야 합니다"
    assert np.allclose(Z_tr.std(axis=0), 1.0), "표준화한 train의 열 표준편차(ddof=0)는 1이어야 합니다"
    assert np.allclose(mean, X_tr.mean(axis=0)) and np.allclose(std, X_tr.std(axis=0))


def test_test_uses_train_statistics():
    X_tr = rng.normal(0.0, 1.0, size=(100, 3))
    X_te = rng.normal(10.0, 5.0, size=(50, 3))          # test는 분포가 다르다
    _, Z_te, mean, std = standardize_split(X_tr, X_te)
    assert np.allclose(mean, X_tr.mean(axis=0)), "평균은 train에서만 구해야 합니다"
    assert np.allclose(Z_te, (X_te - X_tr.mean(axis=0)) / X_tr.std(axis=0)), "test는 train의 통계량으로 변환해야 합니다"
    assert Z_te.mean() > 5, "test의 분포 이동이 표준화 뒤에도 보여야 합니다(test로 다시 맞추면 사라집니다)"


def test_test_does_not_change_statistics():
    X_tr = rng.normal(size=(30, 2))
    _, _, m1, s1 = standardize_split(X_tr, rng.normal(size=(5, 2)))
    _, _, m2, s2 = standardize_split(X_tr, rng.normal(100.0, 1.0, size=(500, 2)))
    assert np.allclose(m1, m2) and np.allclose(s1, s2), "test 데이터가 바뀌어도 통계량은 같아야 합니다"


def test_nan_and_constant_column():
    X_tr = np.array([[1.0, 4.0], [np.nan, 4.0], [3.0, 4.0]])
    X_te = np.array([[2.0, np.nan], [5.0, 6.0]])
    Z_tr, Z_te, mean, std = standardize_split(X_tr, X_te)
    assert np.allclose(mean, [2.0, 4.0]), "평균은 NaN을 빼고 구합니다"
    assert np.allclose(std, [1.0, 1.0]), "상수 열의 std는 1로 둡니다(첫 열의 std는 1)"
    assert np.isnan(Z_tr[1, 0]) and np.isnan(Z_te[0, 1]), "NaN은 NaN으로 남아야 합니다"
    assert np.allclose(Z_te[1], [3.0, 2.0])
    assert not np.any(np.isinf(Z_tr)), "0으로 나누면 안 됩니다"
