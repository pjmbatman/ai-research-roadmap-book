"""5.4 퀴즈(content/05-python-and-computing/04-algorithms-and-complexity/quiz.toml) 정답을 계산으로 검증한다.

시간 대신 비교 횟수·원소 이동처럼 결정적인 양을 센다.
"""
import bisect
import math
import tomllib
from pathlib import Path

QUIZ = Path(__file__).parents[3] / "content/05-python-and-computing/04-algorithms-and-complexity/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


class Key:
    eq_calls = 0

    def __init__(self, v):
        self.v = v

    def __eq__(self, other):
        Key.eq_calls += 1
        return self.v == other.v

    def __hash__(self):
        return hash(self.v)


def count_dedup_list(n):
    Key.eq_calls = 0
    unique = []
    for k in (Key(i) for i in range(n)):
        if k not in unique:
            unique.append(k)
    return Key.eq_calls


def test_membership_list_vs_set():
    items = [Key(i) for i in range(1000)]
    Key.eq_calls = 0
    Key(999) in items
    assert Key.eq_calls == 1000                      # list: 끝까지 훑는다
    s = set(items)
    Key.eq_calls = 0
    Key(999) in s
    assert Key.eq_calls <= 1                         # set: hash로 바로
    assert correct("membership-list-vs-set") == [0]


def test_hidden_quadratic():
    a, b = count_dedup_list(500), count_dedup_list(1000)
    assert a == 500 * 499 // 2 and b == 1000 * 999 // 2
    assert 3.9 < b / a < 4.1                         # n이 2배 → 4배
    assert correct("hidden-quadratic") == [1]


def test_quadratic_scaling():
    assert 2 * (4e4 / 1e4) ** 2 == Q["quadratic-scaling"]["answer"]


def test_list_pop_front():
    assert correct("list-pop-front") == [0]


def test_bisect_comparisons():
    n = 10**6
    calls = 0

    def key(v):
        nonlocal calls
        calls += 1
        return v

    worst = 0
    for target in [0, n // 3, n - 1, 2 * n]:
        calls = 0
        bisect.bisect_left(range(n), target, key=key)
        worst = max(worst, calls)
    assert abs(worst - Q["bisect-comparisons"]["answer"]) <= Q["bisect-comparisons"]["tolerance"]
    assert math.ceil(math.log2(n)) == 20


def test_profile_reading():
    assert correct("profile-reading") == [0]
