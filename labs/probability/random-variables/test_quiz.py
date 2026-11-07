"""11.1 퀴즈(content/11-probability/01-random-variables/quiz.toml) 정답을 계산으로 검증한다."""
import itertools
import tomllib
from pathlib import Path

import numpy as np
from scipy import integrate, stats

QUIZ = Path(__file__).parents[3] / "content/11-probability/01-random-variables/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_random_variable_statements():
    assert correct("random-variable-statements") == [0, 1]


def test_dice_sum_cdf():
    sums = [a + b for a, b in itertools.product(range(1, 7), repeat=2)]
    F4 = np.mean(np.array(sums) <= 4)
    assert F4 == 6 / 36
    assert abs(Q["dice-sum-cdf"]["answer"] - F4) <= Q["dice-sum-cdf"]["tolerance"]


def test_pdf_above_one():
    X = stats.uniform(0, 0.5)
    assert X.pdf(0.2) == 2.0
    assert abs(integrate.quad(X.pdf, 0, 0.5)[0] - 1) < 1e-9     # 전체 적분은 1
    assert abs((X.cdf(0.21) - X.cdf(0.2)) - 0.02) < 1e-12
    assert correct("pdf-above-one") == [1]


def test_interval_from_cdf():
    p = np.exp(-1) - np.exp(-2)
    assert abs(p - (stats.expon.cdf(2) - stats.expon.cdf(1))) < 1e-12
    assert abs(Q["interval-from-cdf"]["answer"] - p) <= Q["interval-from-cdf"]["tolerance"]


def test_mixed_cdf_jump():
    n = 200_000
    r = np.where(rng.random(n) < 0.4, rng.exponential(5.0, n), 0.0)
    assert abs(np.mean(r <= 0.0) - 0.6) < 0.005          # 0에서의 점프
    assert np.mean(r <= 1e-6) - np.mean(r <= 0.0) < 1e-3  # 0 바로 위로는 연속
    assert correct("mixed-cdf-jump") == [0]


def test_monotone_transform_median():
    y = np.exp(rng.normal(0.0, 0.5, size=400_000))
    assert abs(np.median(y) - 1.0) < 0.01
    assert abs(y.mean() - np.exp(0.125)) < 0.01 and y.mean() > 1.05
    Y = stats.lognorm(s=0.5)
    grid = np.linspace(0.05, 3, 20001)
    assert abs(grid[np.argmax(Y.pdf(grid))] - np.exp(-0.25)) < 1e-3
    assert correct("monotone-transform-median") == [1]
