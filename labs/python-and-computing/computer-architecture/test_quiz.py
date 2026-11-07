"""5.5 퀴즈(content/05-python-and-computing/05-computer-architecture/quiz.toml) 정답을 계산·실행으로 검증한다.

시간 측정은 기계마다 다르므로 큰 여유를 둔 대소 관계만 확인한다.
"""
import math
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/05-python-and-computing/05-computer-architecture/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_row_major_traversal():
    A = np.zeros((4000, 4000))
    assert A.strides == (32_000, 8)            # 열 방향(j) 한 칸 = 8 byte, 행 방향(i) 한 칸 = 32,000 byte
    assert np.asfortranarray(A).strides == (8, 32_000)
    assert correct("row-major-traversal") == [0]


def test_float_equality():
    assert (0.1 + 0.2 == 0.3) is False
    assert math.isclose(0.1 + 0.2, 0.3)
    assert repr(0.1 + 0.2) == "0.30000000000000004"
    # float32로 바꾸면 표현 오차가 오히려 커진다(오답 선택지 3)
    assert abs(float(np.float32(0.1)) - 0.1) > 1e-9 > abs(0.1 - 1 / 10)
    assert correct("float-equality") == [0]


def _cpu(_=None):
    s = 0
    for i in range(1_500_000):
        s += i * i
    return s


def _wall(fn, workers, k=4):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fn, range(k)))
    return time.perf_counter() - t0


@pytest.mark.skipif(not getattr(sys, "_is_gil_enabled", lambda: True)(), reason="free-threaded 빌드에서는 GIL이 없다")
def test_gil_cpu_threads():
    seq, par = _wall(_cpu, 1), _wall(_cpu, 4)
    assert par > 0.5 * seq                     # 4배는커녕 2배도 빨라지지 않는다
    assert correct("gil-cpu-threads") == [1]


def test_threads_for_io():
    sleep = lambda _: time.sleep(0.1)
    assert _wall(sleep, 1) > 2 * _wall(sleep, 4)
    assert correct("threads-for-io") == [0, 1]


def test_async_gpu_timing():
    assert correct("async-gpu-timing") == [1]


def test_vector_add_intensity():
    n = 1_000
    flops, nbytes = n, 3 * np.dtype(np.float32).itemsize * n
    q = Q["vector-add-intensity"]
    assert abs(flops / nbytes - q["answer"]) <= q["tolerance"]
