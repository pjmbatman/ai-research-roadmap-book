"""23.3 퀴즈(content/23-research-practice/03-reproducing-papers/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy import stats

QUIZ = Path(__file__).parents[3] / "content/23-research-practice/03-reproducing-papers/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_single_run_halfwidth():
    t = stats.t.ppf(0.975, 9)
    assert abs(t - 2.262) < 5e-4
    half = t * 0.010 * np.sqrt(1 + 1 / 10)
    q = Q["single-run-halfwidth"]
    assert abs(half - q["answer"]) <= q["tolerance"]
    assert abs(t * 0.010 / np.sqrt(10) - 0.007) < 5e-4       # 해설의 평균 구간 반폭


def test_gap_within_seed_noise():
    # seed 표준편차 0.015라면 단일 실행 두 개의 차이가 0.02 이상인 일은 흔하다(약 35%).
    p = 2 * stats.norm.sf(0.02 / (0.015 * np.sqrt(2)))
    assert p > 0.3
    assert correct("gap-not-author-fault") == [0]


def test_factorial_interaction_detectable():
    # 하나씩 바꾸기로는 교호작용을 추정할 수 없다: 2^2 설계에서 교호작용 대비는 네 칸 모두 필요하다.
    y = {(0, 0): 0.0, (1, 0): 1.0, (0, 1): 1.0, (1, 1): 0.0}     # 강한 교호작용, 주효과 0
    main_a = (y[1, 0] + y[1, 1] - y[0, 0] - y[0, 1]) / 2
    inter = (y[0, 0] + y[1, 1] - y[1, 0] - y[0, 1]) / 2
    ofat_a = y[1, 0] - y[0, 0]                                   # 하나씩 바꾸기의 추정
    assert main_a == 0 and inter == -1 and ofat_a == 1
    assert correct("isolate-causes") == [0]


def test_other_answers():
    assert correct("similar-number-different-condition") == [0]
    assert correct("report-contents") == [0, 1, 2]
