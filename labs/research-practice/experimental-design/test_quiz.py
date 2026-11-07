"""23.5 퀴즈(content/23-research-practice/05-experimental-design/quiz.toml) 정답을 계산으로 검증한다."""
import itertools
import math
import tomllib
from pathlib import Path

import numpy as np
from scipy import stats

QUIZ = Path(__file__).parents[3] / "content/23-research-practice/05-experimental-design/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_paired_seed_count():
    n = ((1.96 + 0.84) * 0.012 / 0.01) ** 2
    assert 11 < n < 12
    assert Q["paired-seed-count"]["answer"] == math.ceil(n)
    # 짝짓지 않으면(차이의 sd = sqrt(2) * 0.02) 훨씬 많이 필요하다: 방법당 n = 2 (z sd / delta)^2
    n_unpaired = 2 * ((1.96 + 0.84) * 0.02 / 0.01) ** 2
    assert 60 < n_unpaired < 70


def test_tuning_budget_gap():
    rng = np.random.default_rng(5)

    def tuned(budget):
        q = rng.normal(0.80, 0.02, size=budget)
        v = q + rng.normal(0, 0.01, size=budget)
        return q[np.argmax(v)]

    gap = np.mean([tuned(50) - tuned(5) for _ in range(3000)])
    assert 0.015 < gap < 0.025
    assert correct("tuning-budget") == [0]


def test_structure_vs_recipe_numbers():
    # 본문 셀 3과 같은 가상 실험: 기대값으로 계산
    f = lambda a, r, g: 0.80 + 0.002 * a + 0.020 * r + 0.006 * g + 0.010 * a * r
    total = f(1, 1, 0) - f(0, 0, 0)
    recipe_main = np.mean([f(a, 1, g) - f(a, 0, g) for a, g in itertools.product([0, 1], repeat=2)])
    arch_old_recipe = f(1, 0, 0) - f(0, 0, 0)
    assert round(total, 3) == 0.032 and round(recipe_main, 3) == 0.025 and round(arch_old_recipe, 3) == 0.002
    assert correct("structure-vs-recipe") == [0]


def test_other_answers():
    assert correct("uncontrolled-comparison") == [0]
    assert correct("sensitivity-purpose") == [0, 1]
    assert correct("post-hoc-metric") == [0]
