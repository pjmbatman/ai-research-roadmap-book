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


run_ops = load().run_ops
grad_fn = lambda w: 2 * (w - 3.0)                  # L(w) = (w - 3)^2, 최솟값 w = 3


def test_correct_loop_converges():
    w, _ = run_ops(["zero_grad", "backward", "step"] * 50, w0=0.0, lr=0.1, grad_fn=grad_fn)
    assert np.isclose(w, 3.0, atol=1e-4), "올바른 순서의 SGD는 최솟값 w = 3으로 수렴해야 합니다"


def test_single_step_value():
    w, grad = run_ops(["zero_grad", "backward", "step"], w0=0.0, lr=0.1, grad_fn=grad_fn)
    assert np.isclose(w, 0.6) and np.isclose(grad, -6.0), "w = 0 - 0.1 * (-6) = 0.6, 버퍼는 -6이어야 합니다"


def test_missing_zero_grad_accumulates():
    w, grad = run_ops(["backward", "step"] * 2, w0=0.0, lr=0.1, grad_fn=grad_fn)
    # 1회: g=-6 -> w=0.6 / 2회: g=2(0.6-3)=-4.8, 버퍼 -10.8 -> w = 0.6 + 1.08 = 1.68
    assert np.isclose(grad, -10.8), "zero_grad가 없으면 버퍼에 gradient가 쌓여야 합니다"
    assert np.isclose(w, 1.68)


def test_missing_zero_grad_does_not_converge_cleanly():
    w_ok, _ = run_ops(["zero_grad", "backward", "step"] * 30, 0.0, 0.1, grad_fn)
    w_bad, _ = run_ops(["backward", "step"] * 30, 0.0, 0.1, grad_fn)
    assert abs(w_ok - 3.0) < 1e-2 and abs(w_bad - 3.0) > 0.1, "누적된 버퍼는 최솟값을 지나쳐 진동합니다"


def test_zero_grad_between_backward_and_step():
    w, grad = run_ops(["backward", "zero_grad", "step"] * 10, w0=0.0, lr=0.1, grad_fn=grad_fn)
    assert w == 0.0 and grad is None, "step 직전에 버퍼를 비우면 파라미터가 바뀌지 않습니다"


def test_accumulation_sums_gradients():
    w, grad = run_ops(["zero_grad", "backward", "backward", "step"], 0.0, 0.1, grad_fn)
    assert np.isclose(grad, -12.0) and np.isclose(w, 1.2), "두 번의 backward는 같은 w에서의 gradient 합(-12)입니다"


def test_backward_uses_current_w():
    ops = ["zero_grad", "backward", "step", "backward"]
    w, grad = run_ops(ops, 0.0, 0.1, grad_fn)
    assert np.isclose(grad, -6.0 + 2 * (0.6 - 3.0)), "backward는 갱신된 현재 w에서 gradient를 계산합니다"


def test_unknown_op():
    with pytest.raises(ValueError):
        run_ops(["zero_grad", "forward"], 0.0, 0.1, grad_fn)
