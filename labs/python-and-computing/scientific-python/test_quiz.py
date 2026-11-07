"""5.3 퀴즈(content/05-python-and-computing/03-scientific-python/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

QUIZ = Path(__file__).parents[3] / "content/05-python-and-computing/03-scientific-python/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_which_modify_original():
    snippets = [
        "b = a[1:4]; b[:] = 0",
        "b = a.reshape(2, 3); b *= 2",
        "b = a[[1, 2]]; b[:] = 0",
        "b = a[a > 2]; b[:] = 0",
    ]
    changed = []
    for i, code in enumerate(snippets):
        a = np.arange(6.0)
        exec(code, {"a": a, "np": np})
        if not np.array_equal(a, np.arange(6.0)):
            changed.append(i)
    assert correct("which-modify-original") == changed == [0, 1]


def test_broadcast_column_minus_row():
    rng = np.random.default_rng(0)
    y = rng.normal(size=4)
    y_pred = (y + 0.1)[:, None]
    assert (y_pred - y).shape == (4, 4)
    pairs = np.mean([(p - t) ** 2 for p in y_pred[:, 0] for t in y])
    assert np.isclose(np.mean((y_pred - y) ** 2), pairs)
    assert correct("broadcast-column-minus-row") == [1]


def test_mean_axis_0():
    X = np.arange(12.0).reshape(4, 3)    # (N, D)
    assert X.mean(axis=0).shape == (3,) and np.allclose(X.mean(axis=0), [4.5, 5.5, 6.5])
    assert X.mean(axis=1).shape == (4,)
    assert correct("mean-axis-0") == [0]


def test_int_array_assign():
    a = np.array([1, 2, 3])
    a[0] = 0.9
    assert a[0] == Q["int-array-assign"]["answer"]


def test_why_vectorization_fast():
    assert correct("why-vectorization-fast") == [0]


def test_resample_daily_rows():
    s = pd.Series(np.arange(72.0), index=pd.date_range("2024-07-01", periods=72, freq="h"))
    assert len(s.resample("D").mean()) == Q["resample-daily-rows"]["answer"]
