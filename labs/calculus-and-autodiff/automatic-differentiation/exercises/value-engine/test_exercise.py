import importlib.util
import math
import os
from pathlib import Path

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Value = load().Value


def close(a, b, tol=1e-10):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_forward_values():
    a, b = Value(2.0), Value(-3.0)
    assert close((a + b).data, -1.0), "덧셈의 값이 틀립니다"
    assert close((a * b).data, -6.0), "곱셈의 값이 틀립니다"
    assert close(a.tanh().data, math.tanh(2.0)), "tanh의 값이 틀립니다"
    assert close((a + 1.5).data, 3.5) and close((a * 2).data, 4.0), "숫자와의 연산은 숫자를 Value로 감싸야 합니다"


def test_neuron_gradients():
    x1, x2, w1, w2, b = Value(2.0), Value(-1.0), Value(-0.5), Value(1.5), Value(0.8)
    y = (x1 * w1 + x2 * w2 + b).tanh()
    y.backward()
    s = 1 - math.tanh(2.0 * -0.5 + -1.0 * 1.5 + 0.8) ** 2
    assert close(y.grad, 1.0), "backward는 출력의 grad를 1로 두고 시작해야 합니다"
    assert close(w1.grad, s * 2.0), f"dy/dw1이 틀립니다: {w1.grad} (기대값 {s * 2.0})"
    assert close(w2.grad, s * -1.0), f"dy/dw2가 틀립니다: {w2.grad}"
    assert close(b.grad, s), f"dy/db가 틀립니다: {b.grad}"
    assert close(x1.grad, s * -0.5), f"dy/dx1이 틀립니다: {x1.grad}"


def test_shared_node_accumulates():
    u = Value(3.0)
    z = u * u + u
    z.backward()
    assert close(u.grad, 7.0), f"dz/du = 2u + 1 = 7이어야 합니다. 현재 {u.grad}. gradient를 덮어쓰지 말고 더하세요"


def test_diamond_graph_order():
    x, y = Value(2.0), Value(3.0)
    a = x * y            # a = 6
    c = a * (a + x)      # c = a^2 + a x
    c.backward()
    # dc/da = 2a + x = 14, dc/dx = a + (dc/da) * y = 6 + 42 = 48, dc/dy = (dc/da) * x = 28
    assert close(c.data, 48.0)
    assert close(a.grad, 14.0), f"dc/da = 14여야 합니다. 현재 {a.grad}"
    assert close(x.grad, 48.0), f"dc/dx = 48이어야 합니다. 현재 {x.grad}. 위상 순서를 확인하세요"
    assert close(y.grad, 28.0), f"dc/dy = 28이어야 합니다. 현재 {y.grad}"


def test_constant_operand():
    w = Value(0.5)
    out = (w * 4 + 1.0).tanh()
    out.backward()
    expected = 4 * (1 - math.tanh(3.0) ** 2)
    assert close(w.grad, expected), f"dout/dw = {expected:.6f}이어야 합니다. 현재 {w.grad}"
