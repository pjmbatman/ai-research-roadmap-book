"""30.3 퀴즈(content/30-battery-ai/03-battery-data/quiz.toml) 정답을 계산으로 검증한다.

본문 cells.py를 실행해 그 결과(분할별 MAE, 중도절단, EOL)로 문항의 수치를 확인한다.
"""
import runpy
import tomllib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
QUIZ = Path(__file__).parents[3] / "content/30-battery-ai/03-battery-data/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


@pytest.fixture(scope="module")
def ns():
    return runpy.run_path(str(HERE / "cells.py"))


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_dataset_heterogeneity():
    assert correct("dataset-heterogeneity") == [0]


def test_cycle_split_leak(ns):
    y, life = ns["y"], ns["life"]
    mae = lambda p: np.mean(np.abs(p - y))
    assert round(mae(ns["pred_cyc"])) == 58
    assert round(mae(ns["pred_cell"])) == 171
    assert round(np.mean(np.abs(life.mean() - life))) == 169
    assert "58" in Q["cycle-split-leak"]["prompt"] and "171" in Q["cycle-split-leak"]["prompt"]
    assert correct("cycle-split-leak") == [0]


def test_split_unit_claim():
    assert correct("split-unit-claim") == [0]


def test_censored_drop_bias(ns):
    life, obs = ns["life"], ns["observed"]
    assert (~obs).sum() == 13 and len(life) == 40
    assert round(life.mean()) == 908 and round(life[obs].mean()) == 790
    assert round(np.minimum(life, 1000).mean()) == 858
    assert correct("censored-drop-bias") == [0]


def test_cycling_vs_rpt_capacity(ns):
    fb, q_true, q_cyc, smooth, thr = ns["first_below"], ns["q_true"], ns["q_cyc"], ns["smooth"], ns["thr"]
    assert fb(q_true, thr) == 1291 and fb(q_cyc, thr) == 1252 and fb(smooth, thr) == 1256
    assert correct("cycling-vs-rpt-capacity") == [0]
