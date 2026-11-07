import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load()


def flat(batches):
    return [int(i) for b in batches for i in b]


@pytest.mark.parametrize("n,bs", [(10, 4), (12, 4), (7, 10), (1, 1)])
def test_covers_every_index_once(n, bs):
    b = mod.epoch_batches(n, bs, seed=0, epoch=0)
    assert sorted(flat(b)) == list(range(n)), "한 epoch에 모든 표본을 정확히 한 번씩 방문해야 합니다"
    assert all(len(x) == bs for x in b[:-1]) and 1 <= len(b[-1]) <= bs


def test_drop_last():
    b = mod.epoch_batches(10, 4, seed=0, epoch=0, drop_last=True)
    assert [len(x) for x in b] == [4, 4], "drop_last=True면 크기가 모자란 마지막 batch를 버립니다"
    b = mod.epoch_batches(12, 4, seed=0, epoch=0, drop_last=True)
    assert [len(x) for x in b] == [4, 4, 4], "나누어떨어지면 아무것도 버리지 않습니다"


def test_no_shuffle_is_ordered():
    assert mod.epoch_batches(5, 2, seed=0, epoch=3, shuffle=False) == [[0, 1], [2, 3], [4]]


def test_deterministic_given_seed_and_epoch():
    a = mod.epoch_batches(50, 8, seed=7, epoch=2)
    b = mod.epoch_batches(50, 8, seed=7, epoch=2)
    assert a == b, "같은 (seed, epoch)이면 같은 순서여야 합니다(재현성)"


def test_epochs_differ():
    orders = {tuple(flat(mod.epoch_batches(50, 8, seed=7, epoch=e))) for e in range(5)}
    assert len(orders) == 5, "epoch마다 다른 순서로 섞어야 합니다"
    assert flat(mod.epoch_batches(50, 8, seed=7, epoch=0)) != flat(mod.epoch_batches(50, 8, seed=8, epoch=0))


def test_assign_round_robin():
    batches = [[0], [1], [2], [3], [4]]
    assert mod.assign_to_workers(batches, 2) == [[[0], [2], [4]], [[1], [3]]]
    assert mod.assign_to_workers(batches, 1) == [batches]


def test_interleaving_restores_order():
    batches = mod.epoch_batches(23, 3, seed=1, epoch=0)
    per_worker = mod.assign_to_workers(batches, 3)
    assert len(per_worker) == 3
    restored = [per_worker[i % 3][i // 3] for i in range(len(batches))]
    assert restored == batches, "worker별 목록을 차례로 엮으면 원래 batch 순서가 되어야 합니다"
