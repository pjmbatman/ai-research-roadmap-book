"""23.12 퀴즈(content/23-research-practice/12-research-ethics/quiz.toml)에서 계산으로 확인할 수 있는 정답을 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/23-research-practice/12-research-ethics/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_selective_reporting_mean_below_baseline():
    seeds = np.array([0.81, 0.79, 0.84, 0.78, 0.80])
    baseline = 0.82
    assert round(seeds.mean(), 3) == 0.804          # 해설과 정답 선택지의 평균
    assert seeds.max() > baseline > seeds.mean()     # 최고값만 보면 이기고, 평균은 진다
    assert "0.804" in Q["selective-reporting"]["choices"][1]["text"]
    assert correct("selective-reporting") == [1]
