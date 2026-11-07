"""30.4 퀴즈(content/30-battery-ai/04-state-and-lifetime-prediction/quiz.toml) 정답을 계산으로 검증한다.

본문 cells.py를 실행해 그 결과(SOC 오차, MAPE, coverage)로 문항의 수치를 확인한다.
"""
import runpy
import tomllib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
QUIZ = Path(__file__).parents[3] / "content/30-battery-ai/04-state-and-lifetime-prediction/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


@pytest.fixture(scope="module")
def ns():
    return runpy.run_path(str(HERE / "cells.py"))


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_measurable_inputs():
    assert correct("measurable-inputs") == [0]


def test_coulomb_offset_drift(ns):
    t_h = 10.0
    drift = 0.05 * t_h / 2.5 * 100
    assert np.isclose(drift, Q["coulomb-offset-drift"]["answer"])
    # 본문 cell 1과 같은 방식의 적산으로도 확인
    i = np.zeros(int(t_h * 3600))
    err = ns["coulomb"](i + 0.05, 2.5, 0.5) - ns["coulomb"](i, 2.5, 0.5)
    assert np.isclose(abs(err[-1]) * 100, drift)


def test_ekf_corrects_initial_error(ns):
    e_cc, e_ekf = ns["e_cc"], ns["e_ekf"]
    assert np.all(np.abs(e_cc) > 25)                       # 2시간 내내 25%p 넘게 틀림
    assert np.all(np.abs(e_ekf[60:]) < 1)                  # 1분 뒤부터 1%p 미만
    assert correct("ekf-corrects-initial-error") == [0]


def test_ekf_flat_ocv(ns):
    # 평탄한 곡선에서 OCV 기울기(관측 Jacobian의 SOC 성분)가 몇 배 작다
    s = 0.5
    flat = lambda x: 3.3 + 0.03 * x + 0.04 * np.log((x + 0.02) / (1.02 - x))
    h = 1e-5
    assert (flat(s + h) - flat(s - h)) / (2 * h) < 0.3 * ns["docv"](s)
    assert correct("ekf-flat-ocv") == [0]


def test_split_for_new_protocol(ns):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_predict
    X, y, life = ns["x_var"][:, None], ns["log_life"], ns["life"]
    mape = {k: np.mean(np.abs(np.exp(cross_val_predict(LinearRegression(), X, y, cv=cv)) - life) / life)
            for k, cv in ns["cvs"].items()}
    assert round(mape["셀 단위 5-fold"] * 100, 1) == 14.6
    assert round(mape["프로토콜 단위"] * 100, 1) == 18.0
    assert correct("split-for-new-protocol") == [0]


def test_conformal_condition_coverage(ns):
    hit = ns["conformal_coverage"](ns["cvs"]["셀 단위 5-fold"])
    proto = ns["proto"]
    by = [hit[proto == g].mean() for g in range(8)]
    assert round(hit.mean(), 2) == 0.90 and round(min(by), 2) == 0.70 and max(by) == 1.0
    hit2 = ns["conformal_coverage"](ns["cvs"]["프로토콜 단위"])
    assert round(hit2.mean(), 2) == 0.82
    assert correct("conformal-condition-coverage") == [0]
