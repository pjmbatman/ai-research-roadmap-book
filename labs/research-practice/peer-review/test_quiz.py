"""23.11 퀴즈(content/23-research-practice/11-peer-review/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/23-research-practice/11-peer-review/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    p_o = np.mean(a == b)
    p_e = sum(np.mean(a == c) * np.mean(b == c) for c in np.union1d(a, b))
    return (p_o - p_e) / (1 - p_e)


def test_kappa_from_table():
    # (심사자 1, 심사자 2): 둘 다 채택 15, 둘 다 거절 65, 1만 채택 10, 2만 채택 10
    pairs = [(1, 1)] * 15 + [(0, 0)] * 65 + [(1, 0)] * 10 + [(0, 1)] * 10
    a, b = zip(*pairs)
    k = kappa(a, b)
    q = Q["kappa-from-table"]
    assert abs(k - q["answer"]) <= q["tolerance"]
    assert abs(k - 7 / 15) < 1e-12


def test_calibrated_scores():
    z_x = (5 - 3.5) / 1
    z_y = (7 - 7.5) / 1
    assert z_x > z_y
    assert correct("calibrated-scores") == [1]


def test_chapter_kappa_example_matches_text():
    r1 = [1, 1, 1, 1] + [0] * 16
    r2 = [1, 1, 0, 0, 1, 1] + [0] * 14
    assert abs(kappa(r1, r2) - 0.375) < 1e-12
    assert abs(kappa(r1, [0] * 20)) < 1e-12


def test_single_choice_answers():
    assert correct("self-citation-anonymity") == [0]
    assert correct("constructive-weakness") == [1]
    assert correct("rebuttal-to-misreading") == [1]
    assert correct("reviewer-conflict-of-interest") == [0, 1, 2]
    assert correct("resubmission-after-rejection") == [0, 1, 4]
