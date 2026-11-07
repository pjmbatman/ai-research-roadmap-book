"""8.4 퀴즈(content/08-calculus-and-autodiff/04-automatic-differentiation/quiz.toml) 정답을 계산으로 검증한다.

PyTorch 없이 NumPy와 이 폴더의 engine.py만 쓴다.
"""
import importlib.util
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/08-calculus-and-autodiff/04-automatic-differentiation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

_spec = importlib.util.spec_from_file_location("ad_engine", HERE / "engine.py")
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)
Value = engine.Value


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_reverse_mode_cost():
    # 입력이 많고 출력이 하나인 함수: backward 한 번으로 모든 입력의 gradient가 나온다
    xs = [Value(v) for v in np.linspace(-1, 1, 50)]
    out = sum((x * x for x in xs), Value(0.0))
    out.backward()
    assert np.allclose([x.grad for x in xs], 2 * np.linspace(-1, 1, 50))
    assert correct("reverse-mode-cost") == [0]


def test_forward_mode_tall():
    # forward mode 한 번 = Jacobian의 열 하나(J e_j), reverse mode 한 번 = 행 하나(e_i^T J)
    rng = np.random.default_rng(0)
    J = rng.normal(size=(1000, 3))
    cols = [J @ e for e in np.eye(3)]
    assert np.allclose(np.stack(cols, axis=1), J)        # 3번으로 전체 복원
    assert correct("forward-mode-tall") == [0]


def test_grad_accumulate():
    u = Value(3.0)
    z = u * u + u
    z.backward()
    assert u.grad == Q["grad-accumulate"]["answer"]


def test_backprop_vs_training():
    w = Value(2.0)
    loss = w * w
    loss.backward()
    assert w.data == 2.0 and w.grad == 4.0               # 값은 그대로, gradient만 생긴다
    loss2 = w * w
    loss2.backward()
    assert w.grad == 8.0                                 # 초기화하지 않으면 누적된다
    assert correct("backprop-vs-training") == [0]


def fd_errors(dtype, hs, x0=1.0):
    f = lambda x: np.exp(np.sin(x))
    exact = np.cos(x0) * np.exp(np.sin(x0))
    out = []
    for h in hs:
        x, hh = dtype(x0), dtype(h)
        out.append(abs(float((f(x + hh) - f(x - hh)) / (2 * hh)) - exact) / abs(exact))
    return np.array(out)


def test_fd_float32():
    e32 = fd_errors(np.float32, [1e-6])[0]
    e64 = fd_errors(np.float64, [1e-6])[0]
    assert e32 > 1e-2 and e64 < 1e-8                     # 같은 h에서 float32만 크게 틀린다
    assert correct("fd-float32") == [0]


def test_fd_step_size():
    hs = 10.0 ** -np.arange(1, 13)
    errs = fd_errors(np.float64, hs)
    best = hs[np.argmin(errs)]
    assert 1e-6 <= best <= 1e-4
    assert correct("fd-step-size") == [0]


def test_vanishing_sigmoid():
    sig = lambda a: 1 / (1 + np.exp(-a))
    a = np.linspace(-10, 10, 200001)
    max_deriv = np.max(sig(a) * (1 - sig(a)))
    assert np.isclose(max_deriv, 0.25)
    bound = max_deriv**10
    q = Q["vanishing-sigmoid"]
    assert abs(bound - q["answer"]) <= q["tolerance"]
    # 실제 합성의 미분은 상한보다 작다
    x, d = 0.0, 1.0
    for _ in range(10):
        s = sig(x)
        d *= s * (1 - s)
        x = s
    assert d < bound
