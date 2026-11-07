"""30.6 퀴즈(content/30-battery-ai/06-charging-optimization/quiz.toml) 정답을 계산으로 검증한다."""
import contextlib
import io
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/30-battery-ai/06-charging-optimization/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def cells(upto):
    """cells.py를 cell `upto` 앞까지 실행한 이름공간."""
    src = (HERE / "cells.py").read_text(encoding="utf-8").split(f"# %% cell {upto}")[0]
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(src, ns)
    return ns


def test_cc_time_to_80():
    minutes = (0.8 - 0.1) / 2.0 * 60                       # ΔSOC / C-rate [h]
    assert abs(minutes - Q["cc-time-to-80"]["answer"]) <= Q["cc-time-to-80"]["tolerance"]
    ns = cells(2)
    _, t80, _, _ = ns["cc_cv"](2.0)
    assert abs(t80 / 60 - minutes) < 0.5


def test_cv_start_soc():
    ns = cells(2)
    lo, hi = ns["cc_cv"](0.5), ns["cc_cv"](3.0)
    assert hi[0] < lo[0] - 0.2                             # CV 시작 SOC가 크게 낮아진다
    assert lo[1] / hi[1] > 5.5 and lo[2] / hi[2] < 4.0     # 80% 시간은 ~6배, 종료 시간은 ~3.5배
    assert correct("cv-start-soc") == [0]


def test_pareto_reporting():
    ns = cells(3)
    t, w, feas = ns["t_min"], ns["wear"], ns["feasible"]
    F = np.column_stack([t, w])
    all_front = np.flatnonzero(ns["pareto_mask"](F))       # 제약 위반 후보까지 넣은 front
    assert not np.all(ns["feasible"][all_front])            # 위반 후보가 front에 들어온다
    assert correct("pareto-reporting") == [0, 1]


def test_feasibility_acquisition():
    assert correct("feasibility-acquisition") == [0]


def test_early_prediction_selection_bias():
    # 참값이 같은 후보 50개에 독립 잡음을 더해 최댓값을 고르면, 고른 후보의 관측값은 참값보다 체계적으로 크다
    rng = np.random.default_rng(0)
    obs = 1000 + 90 * rng.normal(size=(2000, 50))          # 수명 1000사이클, 예측 오차 9%
    assert obs.max(axis=1).mean() > 1100
    assert correct("early-prediction-selection-bias") == [0]


def test_sim_to_real_margin():
    assert correct("sim-to-real-margin") == [0, 1]
