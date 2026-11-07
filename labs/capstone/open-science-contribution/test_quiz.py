"""31.2 퀴즈(content/31-capstone/02-open-science-contribution/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/31-capstone/02-open-science-contribution/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def test_leaderboard_noise():
    q = Q["leaderboard-noise"]
    p, n = 0.90, 1000
    se_pp = 100 * np.sqrt(p * (1 - p) / n)            # %p
    assert abs(se_pp - q["answer"]) <= q["tolerance"]
    # 시뮬레이션: 정확도 0.9인 모델을 n = 1000개 표본으로 반복 평가한 점수의 표준편차
    rng = np.random.default_rng(0)
    acc = rng.binomial(n, p, size=50_000) / n          # (50000,)
    assert abs(100 * acc.std() - q["answer"]) <= q["tolerance"]
    # 대표 오답(‰ 단위 9.5, 분산 p(1-p) = 0.09)은 허용 오차 밖이다
    for wrong in (10 * se_pp, p * (1 - p)):
        assert abs(wrong - q["answer"]) > q["tolerance"]
