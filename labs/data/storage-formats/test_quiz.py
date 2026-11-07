"""15.2 퀴즈(content/15-data/02-storage-formats/quiz.toml) 정답을 계산으로 검증한다."""
import io
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

QUIZ = Path(__file__).parents[3] / "content/15-data/02-storage-formats/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_csv_rounding_error():
    v = 3.7 + np.random.default_rng(0).normal(0, 0.01, 100_000)
    back = pd.read_csv(io.StringIO(pd.DataFrame({"v": v}).to_csv(index=False, float_format="%.3f")))["v"]
    err = np.abs(back.to_numpy() - v).max()
    q = Q["csv-rounding-error"]
    assert err <= 0.0005 + 1e-12
    assert abs(err - q["answer"]) <= q["tolerance"]


def test_npy_row_offset(tmp_path):
    X = np.arange(60, dtype=np.float64).reshape(20, 3)
    p = tmp_path / "x.npy"
    np.save(p, X)
    header = p.stat().st_size - X.nbytes
    assert header == 128
    offset = header + 10 * 3 * 8
    raw = p.read_bytes()[offset:offset + 24]
    assert np.array_equal(np.frombuffer(raw, dtype="<f8"), X[10])
    assert offset == Q["npy-row-offset"]["answer"]


def test_streaming_variance_cancellation():
    x = 1.7e9 + np.random.default_rng(0).normal(0, 1.0, 10_000)
    naive = np.mean(x**2) - np.mean(x) ** 2
    assert abs(naive - x.var()) > 0.5                 # 소거 오차로 분산 1 근처의 값을 잃는다
    assert correct("streaming-variance") == [0]


def test_memory_map_lazy(tmp_path):
    p = tmp_path / "x.npy"
    np.save(p, np.arange(10.0))
    mm = np.load(p, mmap_mode="r")
    assert isinstance(mm, np.memmap)
    assert correct("memory-map") == [0]


def test_single_choice_keys():
    assert correct("columnar-partial-read") == [0]
    assert correct("small-files") == [0]
