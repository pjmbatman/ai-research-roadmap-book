"""22.5 퀴즈(content/22-experiment-operations/05-memory-management/quiz.toml) 정답을 계산으로 검증한다.

PyTorch가 필요한 검증은 pytest.importorskip("torch")로 감싼다(CI의 labs 환경에는 torch가 없다).
"""
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/22-experiment-operations/05-memory-management/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_adam_bytes_per_param():
    n = 10**8
    total_gb = n * (4 + 4 + 8) / 1e9
    assert abs(total_gb - Q["adam-bytes-per-param"]["answer"]) <= Q["adam-bytes-per-param"]["tolerance"]


def test_checkpoint_segment():
    n = 64
    best = min(n // k + k for k in range(1, n + 1) if n % k == 0 and k > 1)
    assert best == Q["checkpoint-segment"]["answer"] == 16
    assert n // 64 + 64 > n                       # 구간 하나는 오히려 나쁘다


def test_accumulation_matches_large_batch_without_batchnorm():
    rng = np.random.default_rng(0)
    X, w = rng.normal(size=(64, 3)), rng.normal(size=3)
    y = X @ np.array([1.0, -2.0, 0.5])
    grad = lambda Xb, yb: 2 * Xb.T @ (Xb @ w - yb) / len(yb)       # 평균 MSE의 gradient
    g_full = grad(X, y)
    g_acc = sum(grad(X[i:i + 16], y[i:i + 16]) for i in range(0, 64, 16)) / 4
    assert np.allclose(g_full, g_acc)
    # BatchNorm 통계: micro-batch 평균의 분산은 전체 분산과 다르다
    var_full = X.var(axis=0)
    var_micro = np.mean([X[i:i + 16].var(axis=0) for i in range(0, 64, 16)], axis=0)
    assert not np.allclose(var_full, var_micro)
    assert correct("accumulation-what-changes") == [0, 1]


def test_single_choice_keys():
    assert correct("nvidia-smi-vs-allocated") == [1]
    assert correct("autocast-saves-what") == [1]
    assert correct("oom-batch-halving") == [1]


def test_autocast_keeps_fp32_params_and_state():
    torch = pytest.importorskip("torch")
    from torch import nn
    torch.manual_seed(0)
    m = nn.Linear(8, 8)
    opt = torch.optim.Adam(m.parameters())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = m(torch.randn(4, 8))
    assert out.dtype == torch.bfloat16
    out.float().sum().backward()
    opt.step()
    assert m.weight.dtype == torch.float32 and m.weight.grad.dtype == torch.float32
    assert opt.state[m.weight]["exp_avg"].dtype == torch.float32
