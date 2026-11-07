"""6.6 퀴즈(content/06-research-code-engineering/06-testing-and-debugging/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

QUIZ = Path(__file__).parents[3] / "content/06-research-code-engineering/06-testing-and-debugging/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_allclose_zero_needs_atol():
    actual = np.array([8.4e-16, -1.4e-15])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual, 0.0)                 # 기본 rtol=1e-7, atol=0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual, 0.0, rtol=1e-3)      # rtol을 키워도 실패
    np.testing.assert_allclose(actual, 0.0, atol=1e-12)
    assert correct("allclose-zero-atol") == [0]


def test_duplicated_formula_passes_wrong_spec():
    # 명세가 RMSE인데 구현과 테스트가 함께 제곱근을 빠뜨리면 테스트는 통과한다
    y, y_hat = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    wrong_rmse = lambda a, b: np.mean((a - b) ** 2)
    assert wrong_rmse(y, y_hat) == np.mean((y - y_hat) ** 2)    # 복제 테스트: 통과
    assert not math.isclose(wrong_rmse(y, y_hat), math.sqrt(12.5))  # 독립 기준: 잡힌다
    assert correct("test-duplicates-implementation") == [0]


def first_bad(n, is_bad):
    good, bad = 0, n - 1
    while bad - good > 1:
        mid = (good + bad) // 2
        if is_bad(mid):
            bad = mid
        else:
            good = mid
    return bad


def test_bisect_worst_case_steps():
    n = 1 + 1024                                     # 인덱스 0 = A(통과), 1..1024 = 후보(1024 = B)
    worst = 0
    for target in range(1, n):
        calls = []
        found = first_bad(n, lambda i: calls.append(i) or i >= target)
        assert found == target
        worst = max(worst, len(calls))
    assert worst == Q["bisect-steps"]["answer"] == 10


def test_multiple_choice_keys():
    assert correct("what-to-test-first") == [0, 1, 2]
    assert correct("pdb-on-failure") == [0]
    assert correct("flaky-test-in-ci") == [0]
