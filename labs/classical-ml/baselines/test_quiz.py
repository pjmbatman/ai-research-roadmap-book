"""16.1 퀴즈(content/16-classical-ml/01-baselines/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

QUIZ = Path(__file__).parents[3] / "content/16-classical-ml/01-baselines/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_persistence_for_short_horizon():
    t_min = np.arange(0, 3 * 24 * 60, 10)                          # 사흘, 10분 간격
    temp_c = 22 + 2 * np.sin(2 * np.pi * t_min / (24 * 60)) + rng.normal(0, 0.05, t_min.size)
    train, test = temp_c[: 2 * 144], temp_c[2 * 144:]
    prev = temp_c[2 * 144 - 1: -1]                                  # 10분 전 값
    mae_persist = np.abs(test - prev).mean()
    mae_mean = np.abs(test - train.mean()).mean()
    assert mae_persist < 0.1 * mae_mean
    assert correct("persistence-for-short-horizon") == [1]


def test_skill_score_numeric():
    assert np.isclose(1 - 3.0 / 4.0, Q["skill-score-numeric"]["answer"])
    assert np.isclose(1 - 3.0 / 30.0, 0.90)                         # 해설의 평균 baseline 기준 값


def test_r2_is_skill_against_mean():
    y = rng.normal(size=50)
    pred = y + rng.normal(0, 0.3, 50)
    ss = 1 - np.mean((pred - y) ** 2) / np.mean((y.mean() - y) ** 2)
    assert np.isclose(r2_score(y, pred), ss)
    assert correct("r2-is-weak-skill") == [0]


def test_fair_tuning_budget():
    assert correct("fair-tuning-budget") == [0]


def test_paired_ci_verdict():
    lo, hi = -1.1, 0.3
    verdict = "이긴다" if hi < 0 else "진다" if lo > 0 else "판정 불가"
    assert verdict == "판정 불가"
    assert correct("paired-ci-verdict") == [2]


def test_baseline_wins_signal():
    assert correct("baseline-wins-signal") == [0, 1]
