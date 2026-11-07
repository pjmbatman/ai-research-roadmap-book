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


grad_check = load().grad_check

A = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])


def f(x):
    return 0.5 * x @ A @ x + np.sum(np.sin(x))


def true_grad(x):
    return A @ x + np.cos(x)


def test_correct_gradient_passes():
    x = np.array([0.3, -1.2, 2.0])
    err = grad_check(f, true_grad(x), x)
    assert np.isscalar(err) or np.ndim(err) == 0, "상대오차의 최댓값(스칼라)을 반환해야 합니다"
    assert err < 1e-7, f"올바른 gradient의 상대오차가 너무 큽니다: {err:.1e}"


def test_wrong_gradients_fail():
    x = np.array([0.3, -1.2, 2.0])
    g = true_grad(x)
    assert grad_check(f, 2 * g, x) > 1e-2, "2배 틀린 gradient를 잡아내야 합니다"
    g_bad = g.copy()
    g_bad[1] *= -1
    assert grad_check(f, g_bad, x) > 1e-2, "한 성분의 부호가 틀린 gradient를 잡아내야 합니다"


def test_zero_gradient_component_is_not_nan():
    h = lambda x: x[0] ** 2 + 0.0 * x[1]          # 둘째 성분의 gradient는 정확히 0
    x = np.array([1.5, 2.0])
    err = grad_check(h, np.array([3.0, 0.0]), x)
    assert np.isfinite(err), "gradient가 0인 성분에서 nan/inf가 나왔습니다. 분모에 하한을 두세요"
    assert err < 1e-7


def test_float32_input_is_promoted():
    x32 = np.array([0.3, -1.2, 2.0], dtype=np.float32)
    g = true_grad(x32.astype(np.float64))
    err = grad_check(f, g, x32)
    assert err < 1e-7, f"float32 입력도 float64로 바꿔 차분해야 합니다. 현재 상대오차 {err:.1e}"


def test_input_not_modified():
    x = np.array([0.3, -1.2, 2.0])
    before = x.copy()
    grad_check(f, true_grad(x), x)
    assert np.array_equal(x, before), "입력 x를 제자리에서 바꾸면 안 됩니다"
