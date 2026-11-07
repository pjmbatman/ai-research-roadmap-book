"""29.5 퀴즈(content/29-life-sciences/05-bio-evaluation/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

QUIZ = Path(__file__).parents[3] / "content/29-life-sciences/05-bio-evaluation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def identity(a, b):
    return np.mean(np.array(list(a)) == np.array(list(b)))


def test_greedy_cluster_counterexample():
    """대표 기준 탐욕적 군집에서 다른 군집의 구성원끼리 기준 이상일 수 있다."""
    t = 0.5
    r1 = "AAAAAAAAAA"
    m1 = "AAAAACCCCC"                     # r1과 0.5 → r1 군집
    r2 = "DDDDDCCCCC"                     # r1과 0.0, m1과 0.5지만 대표(r1)와 비교해 새 대표가 된다
    assert identity(m1, r1) >= t and identity(r2, r1) < t
    assert identity(m1, r2) >= t          # 다른 군집의 구성원과 대표가 기준 이상
    # 같은 군집 안에서도 구성원끼리는 기준 미만일 수 있다(추이성 없음)
    m1b = "CCCCCAAAAA"
    assert identity(m1b, r1) >= t and identity(m1, m1b) < t
    assert correct("greedy-cluster-guarantee") == [1]


def test_time_split_homology():
    assert correct("time-split-homology") == [0]


def test_zero_hits_upper_bound():
    n = 50
    upper = 1 - 0.05 ** (1 / n)
    assert abs(upper - Q["zero-hits-upper-bound"]["answer"]) <= Q["zero-hits-upper-bound"]["tolerance"]
    # 단측 95% 상한 = 양측 90% Clopper–Pearson 구간의 상한
    ci = binomtest(0, n).proportion_ci(confidence_level=0.90, method="exact")
    assert np.isclose(ci.high, upper)
    assert np.isclose(3 / n, 0.06)


def test_hit_rate_and_circularity():
    assert correct("hit-rate-report") == [1]
    assert correct("in-silico-circularity") == [1]


def test_reproducibility_record():
    assert correct("reproducibility-record") == [0, 1, 2]
