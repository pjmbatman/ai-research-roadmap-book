"""22.7 퀴즈(content/22-experiment-operations/07-inference-and-deployment/quiz.toml) 정답을 계산으로 검증한다.

PyTorch가 필요한 검증은 pytest.importorskip("torch")로 감싼다(CI의 labs 환경에는 torch가 없다).
"""
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/22-experiment-operations/07-inference-and-deployment/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_int16_accumulation():
    n = max(k for k in range(1, 100) if k * 127 * 127 <= np.iinfo(np.int16).max)
    assert n == Q["int16-accumulation"]["answer"] == 2
    assert np.iinfo(np.int32).max // (127 * 127) == 133144


def test_per_channel_scale():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(2, 64)) * np.array([[0.01], [5.0]])
    s = np.abs(W).max() / 127
    q = np.clip(np.round(W / s), -127, 127)
    assert np.all(q[0] == 0)                                  # 작은 행이 모두 0
    s_row = np.abs(W).max(axis=1, keepdims=True) / 127
    assert np.count_nonzero(np.round(W / s_row)[0]) > 50
    assert correct("per-channel-scale") == [1]


def test_batch_latency_throughput():
    t = lambda B: 4.0 + 0.05 * B                              # 본문의 비용 모형(ms)
    thr = lambda B: B / t(B)
    assert thr(128) > thr(1) and t(128) > t(1)
    assert thr(128) / thr(1) < 128                            # 비례하지 않는다
    assert correct("batch-latency-throughput") == [0, 1]


def test_contract_normalization():
    new = np.array([15.0, 20.0, 25.0])
    z_batch = (new - new.mean()) / new.std()
    z_single = np.array([20.0]) - 20.0                        # 요청 하나: 표준편차 0
    assert not np.allclose(z_batch, (new - 25.0) / 8.66)
    assert np.all(z_single == 0)
    assert correct("contract-normalization") == [1]


def test_single_choice_keys():
    assert correct("export-eval-mode") == [1]
    assert correct("export-speed") == [1]


def test_export_eval_mode_with_torch():
    torch = pytest.importorskip("torch")
    from torch import nn
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Dropout(0.5), nn.Linear(8, 1))
    m(torch.randn(16, 4))
    x = torch.randn(8, 4)
    ep_train = torch.export.export(m.train(), (x,))
    m.eval()
    ep_eval = torch.export.export(m, (x,))
    with torch.no_grad():
        ref = m(x)
        assert torch.allclose(ep_eval.module()(x), ref)
        assert not torch.allclose(ep_train.module()(x), ref)
    with pytest.raises(NotImplementedError):                  # 내보낸 모듈은 eval() 전환을 지원하지 않는다(집필 시점)
        ep_train.module().eval()
