"""23.4 퀴즈(content/23-research-practice/04-research-questions/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy import stats

QUIZ = Path(__file__).parents[3] / "content/23-research-practice/04-research-questions/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_harking_rate():
    exact = 1 - 0.95**10
    q = Q["harking-rate"]
    assert abs(exact - q["answer"]) <= q["tolerance"]
    rng = np.random.default_rng(3)
    z = rng.normal(size=(20000, 10))                 # 효과 없는 하위 그룹 10개의 검정통계량
    sim = np.mean((stats.norm.sf(z) < 0.05).any(axis=1))
    assert abs(sim - exact) < 0.015


def test_inconclusive_rule():
    lo, hi, delta = -0.01, 0.04, 0.02
    success, stop = lo > 0, hi < delta
    assert not success and not stop
    assert correct("inconclusive-result") == [0]


def test_other_answers():
    assert correct("testable-hypothesis") == [0]
    assert correct("contribution-type") == [0]
    assert correct("complexity-vs-contribution") == [0]
