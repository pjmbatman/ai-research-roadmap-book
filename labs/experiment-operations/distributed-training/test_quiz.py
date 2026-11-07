"""22.6 퀴즈(content/22-experiment-operations/06-distributed-training/quiz.toml) 정답을 계산으로 검증한다.

PyTorch가 필요한 검증은 pytest.importorskip("torch")로 감싼다(CI의 labs 환경에는 torch가 없다).
"""
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/22-experiment-operations/06-distributed-training/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def _grad(X, y, w, idx):
    Xb, yb = X[idx], y[idx]
    return 2 * Xb.T @ (Xb @ w - yb) / len(idx)


def test_allreduce_equal_shards():
    rng = np.random.default_rng(0)
    X, y, w = rng.normal(size=(64, 3)), rng.normal(size=64), rng.normal(size=3)
    full = _grad(X, y, w, np.arange(64))
    equal = np.mean([_grad(X, y, w, p) for p in np.split(np.arange(64), 4)], axis=0)
    uneven = np.mean([_grad(X, y, w, p) for p in np.split(np.arange(64), [40, 48, 56])], axis=0)
    assert np.allclose(equal, full) and not np.allclose(uneven, full)
    assert correct("allreduce-equal-shards") == [1]


def test_fsdp_memory():
    assert abs(1e9 * 16 / 8 / 1e9 - Q["fsdp-memory"]["answer"]) <= Q["fsdp-memory"]["tolerance"]


def test_effective_batch():
    assert 32 * 8 * 2 == Q["effective-batch"]["answer"]


def test_rank0_metric():
    rng = np.random.default_rng(1)
    is_correct = rng.random(1001) < 0.8
    W = 4
    total = int(np.ceil(1001 / W)) * W
    idx = np.concatenate([np.arange(1001), np.arange(total - 1001)])
    shards = [idx[r::W] for r in range(W)]
    hits = sum(int(is_correct[s].sum()) for s in shards)
    counts = sum(len(s) for s in shards)
    assert counts == 1004                                   # 중복 3개 포함
    assert abs(hits / counts - is_correct.mean()) < abs(is_correct[shards[0]].mean() - is_correct.mean())
    assert correct("rank0-metric") == [2]


def test_single_choice_keys():
    assert correct("which-parallelism") == [1]
    assert correct("set-epoch") == [1]
    assert correct("ddp-checkpoint") == [0, 2]


def test_set_epoch_with_torch():
    pytest.importorskip("torch")
    from torch.utils.data.distributed import DistributedSampler
    data = list(range(20))
    s = DistributedSampler(data, num_replicas=2, rank=0, shuffle=True, seed=0)
    e0, e0_again = list(s), list(s)                         # set_epoch 없이 두 번
    s.set_epoch(1)
    e1 = list(s)
    assert e0 == e0_again and e0 != e1
    s1 = DistributedSampler(data, num_replicas=2, rank=1, shuffle=True, seed=0)
    assert not set(e0) & set(s1)                            # rank끼리 겹치지 않는다


def test_ddp_state_dict_prefix_with_torch():
    torch = pytest.importorskip("torch")
    import tempfile

    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group("gloo", init_method=f"file://{tempfile.mktemp()}", rank=0, world_size=1)
    try:
        m = DDP(nn.Linear(2, 1))
        assert all(k.startswith("module.") for k in m.state_dict())
        nn.Linear(2, 1).load_state_dict(m.module.state_dict())
    finally:
        dist.destroy_process_group()
    assert torch.__version__
