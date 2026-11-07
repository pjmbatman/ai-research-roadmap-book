"""17.4 퀴즈(content/17-deep-learning-pytorch/04-datasets-and-dataloaders/quiz.toml) 정답을 계산으로 검증한다.

PyTorch가 필요한 검증은 pytest.importorskip("torch")로 감싼다(CI의 labs 환경에는 torch가 없다).
"""
import pickle
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/17-deep-learning-pytorch/04-datasets-and-dataloaders/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_sample_contract():
    assert correct("sample-contract") == [0]


def test_padding_in_loss():
    x = np.zeros((2, 6))
    mask = np.zeros((2, 6), dtype=bool)
    x[0, :2], x[1, :6] = 3.0, 3.0
    mask[0, :2], mask[1, :6] = True, True
    err = (0.0 - x) ** 2
    assert err[mask].mean() == 9.0
    q = Q["padding-in-loss"]
    assert abs(err.mean() - q["answer"]) <= q["tolerance"]


def test_element_vs_sample_mean():
    short, long = np.full(2, 1.0), np.full(8, 4.0)          # 표본별 제곱오차
    element = np.concatenate([short, long]).mean()
    per_sample = np.mean([short.mean(), long.mean()])
    assert element > per_sample                             # 긴 표본(오차 4)의 비중이 크다
    assert correct("element-vs-sample-mean") == [0]


def test_default_collate_varlen():
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError):
        torch.utils.data.default_collate([torch.zeros(3, 2), torch.zeros(5, 2)])
    assert torch.utils.data.default_collate(["a", "b"]) == ["a", "b"]   # 문자열은 list
    assert correct("default-collate-varlen") == [0]


class _RngHolder:
    def __init__(self):
        self.rng = np.random.default_rng(0)


def test_worker_same_random():
    d = _RngHolder()
    copies = [pickle.loads(pickle.dumps(d)) for _ in range(4)]
    assert len({c.rng.normal() for c in copies}) == 1
    assert correct("worker-same-random") == [0]


class _CountingDataset:
    """worker에서 속성을 바꿔도 메인 프로세스에 보이지 않는지 확인한다."""

    def __init__(self):
        self.calls = 0

    def __len__(self):
        return 12

    def __getitem__(self, i):
        self.calls += 1
        return i


def test_num_workers_facts():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    ds = _CountingDataset()

    def order(nw):
        g = torch.Generator().manual_seed(0)
        return [b.tolist() for b in DataLoader(ds, batch_size=3, shuffle=True, num_workers=nw, generator=g)]

    assert order(0) == order(2)                             # 순서가 같다
    ds.calls = 0
    order(2)
    assert ds.calls == 0                                    # worker 안의 변경은 보이지 않는다
    order(0)
    assert ds.calls == 12
    assert correct("num-workers-facts") == [0, 1]
