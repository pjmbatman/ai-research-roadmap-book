"""21.2 퀴즈(content/21-reliability/02-evaluating-predictive-distributions/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/21-reliability/02-evaluating-predictive-distributions/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_improper_absolute_score():
    q = np.linspace(0, 1, 101)
    absolute = 0.7 * (1 - q) + 0.3 * q
    brier = 0.7 * (1 - q) ** 2 + 0.3 * q ** 2
    assert q[np.argmin(absolute)] == 1.0
    assert np.isclose(q[np.argmin(brier)], 0.7)
    texts = [float(c["text"]) for c in Q["improper-absolute-score"]["choices"]]
    assert [texts[i] for i in correct("improper-absolute-score")] == [1.0]


def test_brier_single():
    assert np.isclose((0.8 - 0) ** 2, Q["brier-single"]["answer"])


def test_climatology_calibrated_but_wide():
    x = rng.uniform(0, 1, 50000)
    y = 2 * x + rng.normal(size=x.size) * (0.1 + 0.5 * x)
    past = np.sort(2 * (u := rng.uniform(size=50000)) + rng.normal(size=50000) * (0.1 + 0.5 * u))
    lo, hi = np.quantile(past, [0.05, 0.95])
    inside = (lo < y) & (y < hi)
    assert abs(inside.mean() - 0.90) < 0.01                               # 전체로는 보정
    pit = np.searchsorted(past, y) / past.size
    assert np.allclose(np.histogram(pit, bins=5, range=(0, 1))[0] / y.size, 0.2, atol=0.01)
    assert (hi - lo) > 2 * 1.645 * np.sqrt(np.mean((0.1 + 0.5 * x) ** 2)) * 1.5   # 조건부 예측보다 훨씬 넓다
    assert correct("calibrated-not-sharp") == [0]


def test_base_rate_forecast_has_zero_ece():
    y = (rng.uniform(size=100000) < 0.3).astype(float)
    q = np.full_like(y, 0.3)                                              # 늘 기저율만 예보
    ece = abs(y.mean() - q.mean())                                        # 모든 예측이 한 구간
    assert ece < 0.01
    assert correct("ece-zero-limits") == [0, 1, 2]


def test_interval_score():
    l, u, y, a = 1.0, 3.0, 4.0, 0.1
    s = (u - l) + 2 / a * (l - y) * (y < l) + 2 / a * (y - u) * (y > u)
    assert np.isclose(s, Q["interval-score-numeric"]["answer"])


def test_temperature_keeps_ranking():
    z = rng.normal(0, 3, 1000)
    sig = lambda t: 1 / (1 + np.exp(-t))
    q1, q2 = sig(z), sig(z / 2.5)
    assert np.array_equal(q1 > 0.5, q2 > 0.5)                             # 예측 class 그대로
    assert np.array_equal(np.argsort(q1), np.argsort(q2))                 # 순위(AUC) 그대로
    assert not np.allclose(q1, q2)                                        # 확률 크기는 바뀐다
    assert correct("temperature-keeps-accuracy") == [0]
