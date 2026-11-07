"""16.8 퀴즈(content/16-classical-ml/08-survival-and-causal/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/16-classical-ml/08-survival-and-causal/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def km(time, event):
    t_j = np.unique(time[event])
    d = np.array([np.sum((time == t) & event) for t in t_j])
    n = np.array([np.sum(time >= t) for t in t_j])
    return t_j, np.cumprod(1 - d / n)


def test_censoring_drop_bias():
    life = 1000 * rng.weibull(3.0, 20000)
    stop = rng.uniform(400, 1200, 20000)
    time, event = np.minimum(life, stop), life <= stop
    assert np.median(time[event]) < np.median(life)         # 고장만 쓰면 과소추정
    assert correct("censoring-drop-bias") == [0]


def test_km_by_hand():
    time = np.array([2, 3, 5, 6, 7])
    event = np.array([1, 0, 1, 0, 1], dtype=bool)
    t_j, S = km(time, event)
    s5 = S[np.searchsorted(t_j, 5, side="right") - 1]
    assert np.isclose(s5, 8 / 15)
    assert abs(s5 - Q["km-by-hand"]["answer"]) <= Q["km-by-hand"]["tolerance"]
    # 오답 경로: 고장만 쓰면 1/3, 중도절단=고장이면 2/5
    assert np.isclose(np.mean(time[event] > 5), 1 / 3) and np.isclose(np.mean(time > 5), 2 / 5)


def test_rul_drop_censored():
    assert correct("rul-drop-censored") == [0]


def test_confounded_sign():
    m = 200_000
    Z = rng.normal(30, 7, m)
    T = rng.uniform(size=m) < 1 / (1 + np.exp(-0.3 * (Z - 30)))
    Y = 0.2 * Z - 1.0 * T + rng.normal(0, 0.5, m)
    naive = Y[T].mean() - Y[~T].mean()
    adj = np.linalg.lstsq(np.c_[np.ones(m), T, Z], Y, rcond=None)[0][1]
    assert naive > 0 and abs(adj + 1.0) < 0.02             # 단순 비교는 부호가 틀리고, 보정하면 -1
    assert correct("confounded-sign") == [0]


def test_backdoor_adjustment_set():
    m = 200_000
    Z = rng.normal(30, 7, m)
    T = (rng.uniform(size=m) < 1 / (1 + np.exp(-0.3 * (Z - 30)))).astype(float)
    M = 2.0 - 0.8 * T + rng.normal(0, 0.3, m)
    Y = 0.2 * Z + 1.25 * M + rng.normal(0, 0.5, m)
    C = Y < np.quantile(Y, 0.5) + 1.5 * T
    coef = lambda cols, rows=slice(None): np.linalg.lstsq(np.c_[np.ones(m), T, *cols][rows], Y[rows], rcond=None)[0][1]
    assert abs(coef([Z]) + 1.0) < 0.02                      # {Z}: 총효과
    assert abs(coef([Z, M])) < 0.02                         # {Z, M}: 효과 사라짐
    assert abs(coef([Z], rows=C) + 1.0) > 0.2               # C로 선별: 편향
    assert abs(coef([]) + 1.0) > 0.5                        # 보정 없음: 교란
    assert correct("backdoor-adjustment-set") == [0]


def test_feature_importance_not_causal():
    assert correct("feature-importance-not-causal") == [0]


def test_randomization_removes_confounding():
    assert correct("randomization-removes-confounding") == [0]
