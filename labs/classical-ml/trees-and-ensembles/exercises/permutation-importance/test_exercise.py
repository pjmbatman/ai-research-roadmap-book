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


permutation_importance = load().permutation_importance


def data(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))                          # (n, 3)
    y = 2 * X[:, 0] + rng.normal(0, 0.1, n)
    return X, y


def model(X):                                            # x0만 쓰는 모델
    return 2 * X[:, 0]


def test_shape_and_unused_feature_is_zero():
    X, y = data()
    imp = permutation_importance(model, X, y)
    assert np.shape(imp) == (3,), f"반환 shape은 (p,) = (3,)이어야 합니다. 현재 {np.shape(imp)}"
    assert np.allclose(imp[1:], 0.0), "모델이 쓰지 않는 특징을 섞어도 예측은 그대로이므로 중요도는 정확히 0입니다"


def test_used_feature_magnitude():
    X, y = data(1)
    imp = permutation_importance(model, X, y, n_repeats=10)
    expected = 8 * X[:, 0].var()
    assert abs(imp[0] - expected) < 0.15 * expected, f"x0의 중요도는 약 8·Var(x0) = {expected:.2f}여야 합니다. 현재 {imp[0]:.2f}"


def test_input_not_modified_and_reproducible():
    X, y = data(2)
    X0 = X.copy()
    a = permutation_importance(model, X, y, seed=5)
    assert np.array_equal(X, X0), "입력 X를 직접 섞으면 안 됩니다. 복사본을 쓰세요"
    assert np.allclose(a, permutation_importance(model, X, y, seed=5)), "같은 seed면 같은 결과여야 합니다"


def test_correlated_but_unused_feature():
    X, y = data(3)
    X[:, 2] = X[:, 0] + np.random.default_rng(9).normal(0, 0.05, len(y))   # x2는 x0과 거의 같다(출력과 강한 상관)
    assert abs(np.corrcoef(X[:, 2], y)[0, 1]) > 0.9
    imp = permutation_importance(model, X, y)
    assert imp[2] == 0.0, "출력과 상관이 강해도 모델이 쓰지 않는 특징의 중요도는 0입니다"
