"""17.7 퀴즈(content/17-deep-learning-pytorch/07-trainer-design/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/17-deep-learning-pytorch/07-trainer-design/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_early_stopping_epoch():
    losses, patience = [1.0, 0.8, 0.85, 0.79, 0.80, 0.81], 2
    best, wait, stop = float("inf"), 0, None
    for epoch, v in enumerate(losses, start=1):
        if v < best:
            best, wait = v, 0
        else:
            wait += 1
            if wait >= patience:
                stop = epoch
                break
    assert stop == Q["early-stopping-epoch"]["answer"] == 6


def test_adamw_default_weight_decay():
    torch = pytest.importorskip("torch")
    p = [torch.nn.Parameter(torch.zeros(1))]
    assert torch.optim.Adam(p, lr=1e-2).defaults["weight_decay"] == 0
    assert torch.optim.AdamW(p, lr=1e-2).defaults["weight_decay"] == pytest.approx(0.01)
    assert torch.optim.AdamW(p, lr=1e-2).defaults["lr"] == pytest.approx(1e-2)
    assert correct("adamw-default-weight-decay") == [0]


def test_choice_keys():
    assert correct("trainer-boundary") == [0]
    assert correct("callback-vs-loop-structure") == [0, 1, 2]
    assert correct("eval-script-separation") == [0, 1, 2]
    assert correct("framework-adoption") == [0]
