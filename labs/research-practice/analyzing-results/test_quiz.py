"""23.6 퀴즈(content/23-research-practice/06-analyzing-results/quiz.toml) 정답을 계산으로 검증한다."""
import contextlib
import io
import runpy
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/23-research-practice/06-analyzing-results/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def run_cells():
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(HERE / "cells.py"))


def test_average_hides_subgroup():
    g = run_cells()
    d, cond = g["d"], g["cond"]
    assert round(d.mean(), 3) == -0.112                                  # 전체는 B가 낫다
    gaps = {c: round(d[cond == c].mean(), 3) for c in g["n_by_cond"]}
    assert gaps == {"0 °C": 0.270, "25 °C": -0.186, "45 °C": -0.124}     # 프롬프트의 숫자
    assert round(np.mean(d[cond == "25 °C"] < 0), 2) == 0.63              # 개별 셀 승률은 평균과 다르다
    assert correct("average-hides-subgroup") == [1]


def test_mean_only_win():
    lo, hi = -0.004, 0.018
    assert lo < 0 < hi                                                   # 구간이 0을 포함 → 판정 보류
    assert correct("mean-only-win") == [2]


def holm_count(p, alpha):
    p = np.sort(np.asarray(p))
    k = len(p)
    n = 0
    for j, pj in enumerate(p):
        if pj > alpha / (k - j):
            break
        n += 1
    return n


def test_holm_count():
    p = [0.004, 0.014, 0.020, 0.200]
    assert holm_count(p, 0.05) == Q["holm-count"]["answer"] == 3
    assert sum(x <= 0.05 / 4 for x in p) == 1                           # 해설: Bonferroni는 1개


def test_holm_in_cells_beats_bonferroni():
    g = run_cells()
    p, holm = g["p"], g["holm"]
    assert list(holm) == [True, True, True]
    assert 0.05 / 3 < p[2] < 0.05                                        # 45 °C: Bonferroni 실패, Holm 통과


def test_ablation_alternative():
    assert correct("ablation-alternative") == [0, 1, 2]
    g = run_cells()
    runs = g["runs"]
    gap_a = (runs["A (baseline)"] - runs["B 전체"]).mean()
    gap_big = (runs["A 확대"] - runs["B 전체"]).mean()
    assert 0.55 < 1 - gap_big / gap_a < 0.65                             # 본문: 이득의 약 60%는 용량


def test_null_is_not_absence():
    lo, hi, margin = -0.30, 0.05, 0.05
    assert lo < 0 < hi and lo < -margin                                  # 의미 있는 개선을 배제하지 못함 → 판정 불가
    assert correct("null-is-not-absence") == [1]


def test_negative_result_record():
    assert correct("negative-result-record") == [1]
