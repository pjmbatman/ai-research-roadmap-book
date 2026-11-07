"""17.5 퀴즈(content/17-deep-learning-pytorch/05-training-loop/quiz.toml) 정답을 계산으로 검증한다.

PyTorch가 필요한 검증은 pytest.importorskip("torch")로 감싼다(CI의 labs 환경에는 torch가 없다).
"""
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/17-deep-learning-pytorch/05-training-loop/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def _run(ops, w=0.0, lr=0.1, grad=lambda w: 2 * (w - 3.0)):
    buf = None
    for op in ops:
        if op == "z":
            buf = None
        elif op == "b":
            buf = grad(w) if buf is None else buf + grad(w)
        elif op == "s" and buf is not None:
            w -= lr * buf
    return w


def test_zero_grad_position():
    good_front = _run(["z", "b", "s"] * 40)
    good_after = _run(["b", "s", "z"] * 40)
    between = _run(["b", "z", "s"] * 40)
    once = _run(["z"] + ["b", "s"] * 40)
    assert np.isclose(good_front, 3.0, atol=1e-3) and np.isclose(good_after, 3.0, atol=1e-3)
    assert between == 0.0 and not np.isclose(once, 3.0, atol=1e-3)
    assert correct("zero-grad-position") == [0]


def test_steps_per_epoch():
    micro = math.ceil(2000 / 16)
    assert micro == 125
    assert math.ceil(micro / 8) == Q["steps-per-epoch"]["answer"] == 16


def test_scheduler_per_micro_batch():
    epochs, steps, accum = 10, 50, 4
    total = epochs * steps
    cos = lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total) / total))
    assert cos(total // 4 * accum) == pytest.approx(0.0)   # 1/4 지점에서 이미 바닥
    assert correct("scheduler-per-micro-batch") == [0]


def test_update_line():
    torch = pytest.importorskip("torch")
    m = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    w0 = m.weight.detach().clone()
    m(torch.randn(4, 2)).pow(2).mean().backward()
    assert torch.equal(m.weight.detach(), w0)              # backward는 θ를 바꾸지 않는다
    opt.step()
    assert not torch.equal(m.weight.detach(), w0)
    assert correct("update-line") == [1]


def test_train_loss_vs_eval_loss():
    torch = pytest.importorskip("torch")
    from torch import nn
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(2, 32), nn.Dropout(0.5), nn.Linear(32, 1))
    x = torch.randn(16, 2)
    m.train()
    a = m(x)
    m.eval()
    b = m(x)
    assert not torch.allclose(a, b)                        # train/eval 차이
    with torch.no_grad():
        c = m(x)
    assert torch.equal(b, c)                               # autograd 기록 여부는 값을 바꾸지 않는다
    assert correct("train-loss-vs-eval-loss") == [0, 1]


def test_clip_position():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    before = [p.detach().clone() for p in m.parameters()]
    (m(torch.randn(8, 4)) * 100).pow(2).mean().backward()
    total = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=0.1)
    opt.step()
    step = torch.cat([(p.detach() - b).flatten() for p, b in zip(m.parameters(), before)]).norm()
    assert total > 0.1 and step <= 0.1 + 1e-6              # 갱신 크기(lr=1)가 clipping 한계 안
    assert correct("clip-position") == [0]


def test_end_to_end_meaning():
    torch = pytest.importorskip("torch")
    from torch import nn
    enc, head = nn.Linear(3, 4), nn.Linear(4, 1)
    for p in enc.parameters():
        p.requires_grad_(False)                            # 고정한 인코더
    head(enc(torch.randn(5, 3))).sum().backward()
    assert enc.weight.grad is None and head.weight.grad is not None
    assert correct("end-to-end-meaning") == [0]
