"""17.2 퀴즈(content/17-deep-learning-pytorch/02-tensors-and-autograd/quiz.toml) 정답을 계산으로 검증한다.

PyTorch가 필요한 검증은 pytest.importorskip("torch")로 감싼다(CI의 labs 환경에는 torch가 없다).
"""
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/17-deep-learning-pytorch/02-tensors-and-autograd/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_float64_input():
    torch = pytest.importorskip("torch")
    x = torch.from_numpy(np.ones((2, 3)))
    assert x.dtype == torch.float64
    with pytest.raises(RuntimeError):
        torch.nn.Linear(3, 1)(x)
    torch.nn.Linear(3, 1)(x.float())                    # 수정하면 동작한다
    assert correct("float64-input") == [0]


def test_from_numpy_shared():
    torch = pytest.importorskip("torch")
    a = np.ones(3)
    t, c = torch.from_numpy(a), torch.tensor(a)
    a *= 2
    assert t.tolist() == [2.0, 2.0, 2.0] and c.tolist() == [1.0, 1.0, 1.0]
    assert correct("from-numpy-shared") == [1]


def test_view_after_transpose():
    torch = pytest.importorskip("torch")
    x = torch.randn(2, 5, 3)
    with pytest.raises(RuntimeError):
        x.transpose(1, 2).view(2, -1)
    assert x.transpose(1, 2).reshape(2, -1).shape == (2, 15)
    # NumPy로도 같은 판정: 전치는 C-contiguous가 아니다
    assert not np.zeros((2, 5, 3)).transpose(0, 2, 1).flags["C_CONTIGUOUS"]
    assert correct("view-after-transpose") == [0]


def test_broadcast_grad_value():
    # NumPy: 출력 쪽 gradient(모두 1)를 broadcast 축으로 합한다
    assert np.ones((4, 3)).sum(axis=0, keepdims=True).tolist() == [[4.0, 4.0, 4.0]]
    assert Q["broadcast-grad-value"]["answer"] == 4


def test_broadcast_grad_value_torch():
    torch = pytest.importorskip("torch")
    X, b = torch.randn(4, 3), torch.zeros(1, 3, requires_grad=True)
    (X + b).sum().backward()
    assert torch.equal(b.grad, torch.full((1, 3), float(Q["broadcast-grad-value"]["answer"])))


def test_grad_accumulates():
    torch = pytest.importorskip("torch")
    w = torch.tensor(3.0, requires_grad=True)
    (w**2).backward()
    (w**2).backward()
    assert w.grad.item() == Q["grad-accumulates"]["answer"] == 12


def test_graph_connected():
    torch = pytest.importorskip("torch")
    w, x = torch.randn(3, requires_grad=True), torch.randn(3)
    with torch.no_grad():
        ng = w * x
    results = [w * x, (w * x).detach(), ng, w.clone(), x * 2]
    expected = [i for i, r in enumerate(results) if r.requires_grad]
    assert expected == [0, 3]
    assert correct("graph-connected") == expected


def test_inplace_saved_tensor():
    torch = pytest.importorskip("torch")
    a = torch.randn(3, requires_grad=True)
    s = torch.sigmoid(a)
    s.mul_(2)
    with pytest.raises(RuntimeError, match="inplace"):
        s.sum().backward()
    # 반대로 backward에 출력이 필요 없는 연산 뒤의 in-place는 허용된다
    b = torch.randn(3, requires_grad=True)
    t = b + 1
    t.add_(1)
    t.sum().backward()
    assert torch.equal(b.grad, torch.ones(3))
    assert correct("inplace-saved-tensor") == [0]
