"""17.9 퀴즈(content/17-deep-learning-pytorch/09-debugging-training/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/17-deep-learning-pytorch/09-debugging-training/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_initial_loss_value():
    logits = np.zeros((7, 10))
    y = np.array([0, 0, 0, 1, 2, 9, 9])                       # class 비율과 무관
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    loss = -logp[np.arange(7), y].mean()
    assert loss == pytest.approx(math.log(10))
    assert loss == pytest.approx(Q["initial-loss-value"]["answer"], abs=Q["initial-loss-value"]["tolerance"])


def test_overfit_one_batch_limits():
    torch = pytest.importorskip("torch")
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(0)
    x, y = torch.randn(16, 10), torch.randint(0, 5, (16,))

    def final_loss(head, labels, lr=1e-2):
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 5), *head)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(300):
            loss = nn.functional.cross_entropy(model(x), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

    assert final_loss([], y) < 0.01
    assert final_loss([nn.Softmax(dim=1)], y) > 0.5           # softmax 두 번: 잡는다
    assert final_loss([], y, lr=0.0) > 1.0                    # 업데이트 없음: 잡는다
    assert final_loss([], y[torch.randperm(16)]) < 0.01       # 어긋난 label: 놓친다
    assert correct("overfit-one-batch-limits") == [0, 1]


def test_choice_keys():
    assert correct("nan-first-response") == [0]
    assert correct("suspiciously-good") == [0]
    assert correct("one-change-at-a-time") == [0]
