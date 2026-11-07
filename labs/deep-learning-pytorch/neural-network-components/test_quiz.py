"""17.1 퀴즈(content/17-deep-learning-pytorch/01-neural-network-components/quiz.toml) 정답을 계산으로 검증한다.

PyTorch가 필요한 검증은 pytest.importorskip("torch")로 감싼다(CI의 labs 환경에는 torch가 없다).
"""
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/17-deep-learning-pytorch/01-neural-network-components/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_param_count():
    sizes = [20, 64, 64, 3]
    n = sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))
    assert n == Q["param-count"]["answer"] == 5699


def test_param_count_torch():
    torch = pytest.importorskip("torch")
    from torch import nn
    m = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    assert sum(p.numel() for p in m.parameters()) == Q["param-count"]["answer"]


def test_he_init_std():
    n, depth = 512, 10
    h = rng.normal(size=(512, n))
    for s in [np.sqrt(2 / n)]:
        x = h
        for _ in range(depth):
            x = np.maximum(0, x @ (rng.normal(size=(n, n)) * s))
        # He: 두 번째 모멘트가 입력과 같은 크기로 유지된다
        assert 0.5 < np.mean(x**2) < 2.0
    x = h
    for _ in range(depth):
        x = np.maximum(0, x @ (rng.normal(size=(n, n)) * np.sqrt(1 / n)))
    assert np.mean(x**2) < 0.01                       # sqrt(1/n): 층마다 절반 → 2^-10 수준
    assert np.isclose(np.sqrt(2 / 512), 0.0625)
    assert correct("he-init-std") == [2]


def _bn_train(x):
    return (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5)


def _ln(x):
    return (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5)


def _rms(x):
    return x / np.sqrt((x**2).mean(1, keepdims=True) + 1e-6)


def test_norm_batch_dependence():
    x = rng.normal(size=(8, 4)) * 3 + 5
    rm, rv = rng.normal(size=4), rng.uniform(1, 2, size=4)
    bn_eval = lambda z: (z - rm) / np.sqrt(rv + 1e-5)
    fns = [_bn_train, bn_eval, _ln, _rms]             # 선택지 순서
    depends = [not np.allclose(f(x)[:4], f(x[:4])) for f in fns]
    expected = [i for i, d in enumerate(depends) if d]
    assert expected == [0]
    assert correct("norm-batch-dependence") == expected


def test_eval_mode_forgotten():
    torch = pytest.importorskip("torch")
    from torch import nn
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 1))
    for _ in range(5):                                # running 통계를 조금 쌓는다
        m(torch.randn(32, 4))
    x = torch.randn(256, 4)
    with torch.no_grad():
        m.train()
        a, b = m(x), m(x)
        assert not torch.allclose(a, b)                # dropout: 같은 입력에서 결과가 바뀐다
        m.eval()
        big, small = m(x), torch.cat([m(x[i:i + 8]) for i in range(0, 256, 8)])
        assert torch.allclose(big, small, atol=1e-5)   # eval: batch 크기와 무관
    assert correct("eval-mode-forgotten") == [0]


def test_residual_jacobian():
    D = 16
    W = rng.normal(size=(D, D)) * 1e-3                 # 아주 작은 블록
    h = rng.normal(size=D)
    a = np.tanh(W @ h)
    J_plain = (1 - a**2)[:, None] * W
    J_res = np.eye(D) + J_plain
    g = np.ones(D)
    assert np.linalg.norm(J_plain.T @ g) < 1e-2 * np.linalg.norm(g)
    assert np.linalg.norm(J_res.T @ g) > 0.9 * np.linalg.norm(g)
    assert correct("residual-jacobian") == [0]


def test_dropout_scale():
    p = 0.2
    mask = rng.random(200000) >= p
    assert np.isclose((mask / (1 - p)).mean(), 1.0, atol=0.01)
    q = Q["dropout-scale"]
    assert abs(1 / (1 - p) - q["answer"]) <= q["tolerance"]


def test_dropout_scale_torch():
    torch = pytest.importorskip("torch")
    d = torch.nn.Dropout(0.2)
    y = d(torch.ones(10000))
    assert torch.isclose(y.max(), torch.tensor(Q["dropout-scale"]["answer"]))


def test_classification_head():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    logits = torch.randn(64, 5, requires_grad=True)
    y = torch.randint(0, 5, (64,))
    ce = torch.nn.CrossEntropyLoss()
    ref = torch.nn.functional.nll_loss(torch.log_softmax(logits, 1), y)
    assert torch.allclose(ce(logits, y), ref)          # logit을 받는다
    double = ce(torch.softmax(logits, 1), y)           # softmax 두 번
    assert not torch.allclose(double, ref)
    (g1,) = torch.autograd.grad(ce(logits, y), logits)
    (g2,) = torch.autograd.grad(double, logits)
    assert g2.norm() < g1.norm()                       # gradient가 약해진다
    assert correct("classification-head") == [0]
