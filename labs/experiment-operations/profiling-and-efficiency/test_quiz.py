"""22.4 퀴즈(content/22-experiment-operations/04-profiling-and-efficiency/quiz.toml) 정답을 계산으로 검증한다."""
import threading
import time
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/22-experiment-operations/04-profiling-and-efficiency/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_amdahl():
    data, comp = 0.4, 0.6
    speedup = (data + comp) / (data + comp / 2)
    q = Q["amdahl"]
    assert abs(speedup - q["answer"]) <= q["tolerance"]
    assert np.isclose(1 / data, 2.5)                             # 계산을 무한히 빠르게 해도 상한
    assert np.isclose((data + comp) / max(data, comp / 2), 2.5)  # 겹치게 하면 상한에 닿는다


def test_async_timing_without_sync():
    """비동기 실행을 스레드로 흉내 낸다: 동기화 없이 재면 일을 맡기는 시간만 잰다."""
    t0 = time.perf_counter()
    job = threading.Thread(target=time.sleep, args=(0.2,))
    job.start()
    t_launch = time.perf_counter() - t0
    job.join()                                                  # synchronize에 해당
    t_sync = time.perf_counter() - t0
    assert t_launch < 0.1 < t_sync
    assert correct("sync-timing") == [0]


def test_choice_keys():
    assert correct("warmup-first-step") == [0]
    assert correct("low-util-diagnosis") == [0, 1, 2]
    assert correct("speed-vs-accuracy") == [0]
    assert correct("batch-throughput") == [0]
