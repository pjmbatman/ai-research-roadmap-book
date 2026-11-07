"""6.2 퀴즈(content/06-research-code-engineering/02-naming-and-style/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/06-research-code-engineering/02-naming-and-style/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_energy_units():
    voltage_v, current_a, duration_min = 3.7, 2.0, 30.0
    energy_wh = voltage_v * current_a * (duration_min / 60.0)
    wrong_wh = voltage_v * current_a * duration_min / 3600.0     # 분을 초로 착각
    q = Q["energy-units"]
    assert abs(energy_wh - q["answer"]) <= q["tolerance"]
    assert abs(wrong_wh - q["answer"]) > q["tolerance"]
    assert np.isclose(wrong_wh * 60, energy_wh)


def test_broadcast_shape():
    n = 4
    err = np.zeros((n, 1)) - np.zeros(n)
    assert err.shape == (n, n)
    assert correct("broadcast-shape") == [2]


def test_forecast_window_leak():
    series = np.arange(20.0)
    t0, lookback, horizon = 10, 4, 3
    x = series[t0 - lookback + 1 : t0 + 1]
    y_bad = series[t0 : t0 + horizon]
    y_good = series[t0 + 1 : t0 + 1 + horizon]
    assert len(x) == lookback                                   # 선택지 1은 틀림
    assert y_bad[0] == x[-1]                                    # 첫 목표 = 마지막 입력
    assert len(np.intersect1d(x, y_good)) == 0
    assert correct("forecast-window-leak") == [0]


def test_other_answers():
    assert correct("names-to-fix") == [0, 1]
    assert correct("formatter-vs-linter") == [1]
    assert correct("precommit-needs-ci") == [0]
