import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ring_all_reduce = load().ring_all_reduce


@pytest.mark.parametrize("W,n", [(2, 8), (3, 9), (4, 12), (8, 64)])
def test_every_worker_gets_sum(W, n):
    rng = np.random.default_rng(W * 100 + n)
    arrays = [rng.normal(size=n) for _ in range(W)]
    results, _ = ring_all_reduce(arrays)
    assert len(results) == W, "worker마다 결과가 하나씩 있어야 합니다"
    for r in results:
        assert np.allclose(r, np.sum(arrays, axis=0)), "모든 worker의 결과가 전체 합과 같아야 합니다"


@pytest.mark.parametrize("W,n", [(2, 8), (4, 12), (8, 64)])
def test_bandwidth_optimal(W, n):
    arrays = [np.ones(n) * i for i in range(W)]
    _, sent = ring_all_reduce(arrays)
    expected = 2 * (W - 1) / W * arrays[0].nbytes
    for i, b in enumerate(sent):
        assert b == pytest.approx(expected), (
            f"worker {i}가 {b} byte를 보냈습니다. ring all-reduce에서는 모든 worker가 "
            f"2(W-1)/W × 배열 크기 = {expected:.0f} byte를 보내야 합니다")


def test_inputs_unchanged():
    rng = np.random.default_rng(0)
    arrays = [rng.normal(size=6) for _ in range(3)]
    copies = [a.copy() for a in arrays]
    ring_all_reduce(arrays)
    for a, c in zip(arrays, copies):
        assert np.array_equal(a, c), "입력 배열을 바꾸면 안 됩니다(복사해서 쓰세요)"


def test_results_independent():
    results, _ = ring_all_reduce([np.ones(4), np.ones(4)])
    results[0][0] = 100.0
    assert results[1][0] == 2.0, "worker마다 별도의 배열을 돌려줘야 합니다"


def test_single_worker():
    results, sent = ring_all_reduce([np.arange(5.0)])
    assert np.allclose(results[0], np.arange(5.0))
    assert sent == [0], "worker가 하나면 보낼 것이 없습니다"


def test_uneven_chunks():
    rng = np.random.default_rng(1)
    arrays = [rng.normal(size=10) for _ in range(4)]      # 10은 4로 나누어떨어지지 않는다
    results, sent = ring_all_reduce(arrays)
    for r in results:
        assert np.allclose(r, np.sum(arrays, axis=0))
    assert max(sent) <= 2 * arrays[0].nbytes, "worker 하나가 보내는 양은 배열 크기의 2배를 넘지 않아야 합니다"
