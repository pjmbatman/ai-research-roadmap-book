"""17.8 퀴즈(content/17-deep-learning-pytorch/08-dashboards/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/17-deep-learning-pytorch/08-dashboards/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_ema_without_debias():
    s, w = 0.0, 0.9
    for x in (10.0, 10.0):
        s = w * s + (1 - w) * x
    assert s == pytest.approx(Q["ema-without-debias"]["answer"])
    assert s / (1 - w ** 2) == pytest.approx(10.0)              # 보정하면 원래 값


def test_dashboard_train_loss_is_higher():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 5))
    y = X @ np.array([1.0, -1.0, 0.5, 0.0, 2.0]) + 0.5 * rng.normal(size=2000)
    w, losses = np.zeros(5), []
    for i in range(0, 2000, 20):
        r = X[i:i + 20] @ w - y[i:i + 20]
        losses.append(np.mean(r ** 2))
        w -= 0.02 * 2 * X[i:i + 20].T @ r / 20
    assert np.mean(losses) > 2 * np.mean((X @ w - y) ** 2)
    assert correct("dashboard-train-loss") == [0]


def test_update_ratio_signals_divergence():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    x = torch.randn(256, 8)
    y = x[:, :1] * 2 - x[:, 1:2]
    ratios = []
    for _ in range(4):
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad(); loss.backward()
        before = torch.cat([p.detach().flatten() for p in model.parameters()])
        opt.step()
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        ratios.append(((after - before).norm() / before.norm()).item())
    assert ratios[-1] > 1.0                                    # 손실이 inf가 되기 전에 비율이 1을 넘는다
    assert correct("what-to-log-before-divergence") == [0, 1, 2]


def test_choice_keys():
    assert correct("tensorboard-remote-access") == [0]
    assert correct("overfitting-pattern") == [0]
