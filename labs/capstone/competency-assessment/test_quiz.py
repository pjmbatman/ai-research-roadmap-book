"""31.3 퀴즈(content/31-capstone/03-competency-assessment/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/31-capstone/03-competency-assessment/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def test_evidence_rule_mean():
    q = Q["evidence-rule-mean"]
    claimed = np.array([4, 3, 3, 2])        # (4,)
    evidenced = np.array([4, 2, 1, 2])      # (4,)
    now = np.minimum(claimed, evidenced)    # (4,)
    assert abs(now.mean() - q["answer"]) <= q["tolerance"]
    # 대표 오답: 근거 규칙을 적용하지 않은 평균 3.0, 근거 없는 칸을 한 단계만 내린 평균 2.5
    one_step = np.where(evidenced < claimed, claimed - 1, claimed)
    for wrong in (claimed.mean(), one_step.mean()):
        assert abs(wrong - q["answer"]) > q["tolerance"]


def test_mean_hides_lowest():
    """본문 cell 1의 수치에서, 축 평균이 가장 낮은 축과 가장 낮은 항목이 속한 축이 다르다."""
    axis = np.array(["지식", "지식", "기술", "기술", "기술", "연구 습관", "연구 습관"])
    now = np.minimum(np.array([4, 3, 4, 3, 4, 3, 3]), np.array([4, 3, 4, 1, 4, 3, 2]))
    means = {a: now[axis == a].mean() for a in ["지식", "기술", "연구 습관"]}
    assert min(means, key=means.get) == "연구 습관"
    assert axis[np.argmin(now)] == "기술"
    correct = [c["text"] for c in Q["mean-hides-lowest"]["choices"] if c.get("correct")]
    assert len(correct) == 1 and "실패 원인 분리" in correct[0]
