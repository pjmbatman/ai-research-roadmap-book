"""21.6 퀴즈(content/21-reliability/06-decision-making/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import norm

QUIZ = Path(__file__).parents[3] / "content/21-reliability/06-decision-making/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)
sig = lambda z: 1 / (1 + np.exp(-z))


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_bayes_threshold():
    p = np.linspace(0, 1, 100001)
    alarm_better = (1 - p) * 1.0 < p * 9.0
    assert np.isclose(p[np.argmax(alarm_better)], Q["bayes-threshold-numeric"]["answer"], atol=1e-4)


def test_threshold_needs_calibration():
    s = rng.normal(size=100000)
    p = sig(-3.0 + 1.5 * s)
    y = rng.uniform(size=s.size) < p
    cost = lambda q, thr: np.mean(1.0 * ((q > thr) & ~y) + 20.0 * ((q <= thr) & y))
    bad = sig(2.5 * (-3.0 + 1.5 * s) + 3.0)               # 순위는 같고 보정만 틀림
    thr = 1 / 21
    grid = np.linspace(0.001, 0.6, 600)
    best_bad = min(cost(bad, g) for g in grid)
    assert cost(p, thr) < cost(bad, thr)                   # 보정이 틀리면 이론 threshold가 나쁘다
    assert abs(best_bad - min(cost(p, g) for g in grid)) < 0.005   # 순위가 같으면 최적 비용이 거의 같다(격자 차이만)
    assert correct("threshold-needs-calibration") == [0]


def test_quantile_decision():
    mu, sd = rng.uniform(5, 15, 50000), rng.uniform(0.5, 3, 50000)
    heat = mu + sd * rng.normal(size=mu.size)
    cost = lambda cap: np.mean(9 * np.maximum(heat - cap, 0) + np.maximum(cap - heat, 0))
    taus = np.linspace(0.5, 0.99, 50)
    best = taus[np.argmin([cost(mu + norm.ppf(t) * sd) for t in taus])]
    assert abs(best - 0.9) < 0.02
    assert cost(mu + norm.ppf(0.9) * sd) < cost(mu + norm.ppf(0.9) * sd.mean()) < cost(mu)
    assert correct("quantile-decision") == [0]


def test_selective_prediction_concentrates_abstention():
    n = 50000
    gb = rng.uniform(size=n) < 0.2
    p1 = sig(rng.normal(size=n) * np.where(gb, 0.8, 2.5))
    conf = np.maximum(p1, 1 - p1)
    answer = conf >= np.quantile(conf, 0.4)
    assert answer[gb].mean() < 0.5 * answer[~gb].mean()
    assert correct("coverage-risk") == [0]


def test_chow_rule():
    d = 0.2
    conf = np.linspace(0.5, 1.0, 50001)
    reject_better = d < 1 - conf
    assert np.isclose(conf[reject_better].max(), Q["chow-reject"]["answer"], atol=1e-4)


def test_default_threshold_cases():
    # 비용이 같고 보정되어 있으면 0.5가 최적
    s = rng.normal(size=100000)
    p = sig(1.5 * s)
    y = rng.uniform(size=s.size) < p
    grid = np.linspace(0.05, 0.95, 91)
    costs = [np.mean(((p > g) & ~y) | ((p <= g) & y)) for g in grid]
    assert abs(grid[np.argmin(costs)] - 0.5) < 0.05
    # label shift(고장률 다섯 배)면 보정이 깨져 0.5가 최적이 아니다
    p_shift = sig(1.5 * s + np.log(5))                      # 운영 환경의 참 확률(사전 odds 5배)
    y2 = rng.uniform(size=s.size) < p_shift
    costs2 = [np.mean(((p > g) & ~y2) | ((p <= g) & y2)) for g in grid]
    assert grid[np.argmin(costs2)] < 0.4
    assert correct("default-threshold") == [0, 1, 2]
