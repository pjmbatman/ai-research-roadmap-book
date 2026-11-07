"""3.6 퀴즈(content/03-environments/06-environment-diagnosis/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

QUIZ = Path(__file__).parents[3] / "content/03-environments/06-environment-diagnosis/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_bisect_steps():
    # 64개 후보 중 처음 실패한 커밋을 찾는 이분 탐색을 실제로 돌려 최악의 시험 횟수를 센다
    n = 64
    worst = 0
    for first_bad in range(n):
        lo, hi, tests = 0, n - 1, 0
        while lo < hi:
            mid = (lo + hi) // 2
            tests += 1
            if mid >= first_bad:          # mid에서 실패
                hi = mid
            else:
                lo = mid + 1
        assert lo == first_bad
        worst = max(worst, tests)
    assert worst == math.log2(n) == Q["bisect-steps"]["answer"] == 6


def test_other_answers():
    assert correct("missing-shared-library") == [0]
    assert correct("no-kernel-image") == [0]
    assert correct("issue-report-contents") == [0, 1, 2]
    assert correct("minimal-reproduction") == [0]
    assert correct("rollback-prerequisites") == [0, 1, 2]


def test_every_wrong_choice_has_feedback():
    for q in Q.values():
        for c in q.get("choices", []):
            assert c.get("correct") or c.get("feedback"), f"{q['id']}: 오답에 feedback이 없다"
