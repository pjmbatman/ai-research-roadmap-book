"""27.2 퀴즈(content/27-earth-and-climate/02-weather-forecasting/quiz.toml) 정답을 계산으로 검증한다."""
import contextlib
import io
import re
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/27-earth-and-climate/02-weather-forecasting/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

# 본문 셀을 실행해 그 결과로 문항의 근거를 확인한다(약 3초).
NS = {"__name__": "cells"}
with contextlib.redirect_stdout(io.StringIO()):
    exec((HERE / "cells.py").read_text(encoding="utf-8"), NS)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_rollout_instability():
    fc0 = NS["rollout"](NS["fit"](0.0), NS["truth"][:, 0])
    fc7 = NS["rollout"](NS["fit"](0.7), NS["truth"][:, 0])
    with np.errstate(all="ignore"):
        r0, r7 = NS["rmse"](fc0), NS["rmse"](fc7)
    assert r0[1] < r7[1]                                        # 1-step은 잡음 없는 모델이 더 정확
    assert (~np.isfinite(fc0[:, -1]).all(-1)).sum() > 50        # 그런데 많이 발산
    assert np.isfinite(fc7).all()
    assert np.isfinite(NS["control"]).all()                     # 참 동역학은 발산하지 않는다
    assert correct("rollout-instability") == [0]


def test_shared_local_weights():
    # Lorenz-96은 위치 불변: 상태를 돌리면(roll) 변화도 똑같이 돌아간다
    x = NS["truth"][0, 0]
    assert np.allclose(NS["tend"](np.roll(x, 5)), np.roll(NS["tend"](x), 5))
    assert correct("shared-local-weights") == [0]


def test_ensemble_mean_smooth():
    rm, rc, rclim = NS["rmse"](NS["mean"])[40], NS["rmse"](NS["control"])[40], NS["r_clim"][40]
    assert rm < rclim < rc
    # 평균장은 멤버보다 매끄럽다: 이웃 격자점 차분의 분산이 작다
    d = lambda f: np.diff(f[:, 40], axis=-1).var()
    assert d(NS["mean"]) < 0.5 * d(NS["members"][0])
    assert correct("ensemble-mean-smooth") == [0]


def test_two_member_crps():
    x, y = np.array([1.0, 3.0]), 2.5
    crps = np.abs(x - y).mean() - 0.5 * np.abs(x[:, None] - x[None]).mean()
    q = Q["two-member-crps"]
    assert abs(crps - q["answer"]) <= q["tolerance"]


def test_skill_score_value():
    q = Q["skill-score-value"]
    assert abs((1 - 2.0**2 / 4.0**2) - q["answer"]) <= q["tolerance"]
    assert abs((1 - 2.0 / 4.0) - q["answer"]) > q["tolerance"]


def test_persistence_long_lead():
    ratio = NS["r_pers"][40] / NS["r_clim"][40]
    q = Q["persistence-long-lead"]
    assert abs(np.sqrt(2) - q["answer"]) <= q["tolerance"]
    assert 1.3 < ratio < 1.6                                    # Lorenz-96에서도 근사적으로 성립


def test_time_split_leakage():
    # 무작위 분할의 누수: 6시간 이웃 상태는 매우 비슷하다(1-step 지속성 RMSE << 기후값 RMSE)
    assert NS["r_pers"][1] < 0.3 * NS["r_clim"][1]
    assert correct("time-split-leakage") == [0, 1]


def test_quiz_mentions_match_cells():
    text = QUIZ.read_text(encoding="utf-8")
    assert "99개가 발산" in text and re.search(r"0\.15", text)
