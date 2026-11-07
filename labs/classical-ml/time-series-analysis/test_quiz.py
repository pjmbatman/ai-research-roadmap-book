"""16.7 퀴즈(content/16-classical-ml/07-time-series-analysis/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/16-classical-ml/07-time-series-analysis/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def acf(x, k):
    x = x - x.mean()
    return np.sum(x[:-k] * x[k:]) / np.sum(x * x)


def test_acf_seasonal_lag():
    t = np.arange(24 * 60)
    y = 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, len(t))
    assert acf(y, 12) < -0.5 and acf(y, 24) > 0.8          # 반 주기 음수, 한 주기 양수
    assert correct("acf-seasonal-lag") == [0]


def test_ar1_forecast_value():
    e, out = 2.0, None
    for _ in range(3):
        e = 0.8 * e
    out = e
    assert np.isclose(out, 0.8 ** 3 * 2.0)
    assert abs(out - Q["ar1-forecast-value"]["answer"]) <= Q["ar1-forecast-value"]["tolerance"]


def test_ses_weight():
    alpha = 0.3
    y = np.zeros(10)
    y[-3] = 1.0                                             # y_{t-2}만 1
    level = 0.0
    for v in y:
        level = alpha * v + (1 - alpha) * level
    assert np.isclose(level, alpha * (1 - alpha) ** 2)
    assert abs(level - Q["ses-weight"]["answer"]) <= Q["ses-weight"]["tolerance"]


def test_filter_vs_smoother():
    assert correct("filter-vs-smoother") == [0]


def test_persistence_horizon():
    # 원점이 23시로 고정된 순수 사인 주기에서 persistence 오차: h=24에서 0, h=6에서 최대 근처
    hour0 = 23
    err = lambda h: abs(3 * np.sin(2 * np.pi * (hour0 + h) / 24) - 3 * np.sin(2 * np.pi * hour0 / 24))
    assert err(24) < 1e-9 and err(6) > err(12) > err(1)
    assert correct("persistence-horizon") == [0]


def test_future_features():
    assert correct("future-features") == [0, 1, 2]


def test_random_split_ar():
    assert correct("random-split-ar") == [0]
