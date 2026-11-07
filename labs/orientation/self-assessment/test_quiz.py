"""0.4 퀴즈(content/00-orientation/04-self-assessment/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/00-orientation/04-self-assessment/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def test_low_rank_product():
    ranks = {np.linalg.matrix_rank(rng.normal(size=(5, 2)) @ rng.normal(size=(2, 4))) for _ in range(20)}
    assert ranks == {Q["low-rank-product"]["answer"]}


def test_standard_error():
    q = Q["standard-error"]
    sigma, n = 2.0, 25
    assert abs(sigma / np.sqrt(n) - q["answer"]) <= 1e-12
    means = rng.normal(0.0, sigma, size=(50_000, n)).mean(axis=1)   # (50000,)
    assert abs(means.std() - q["answer"]) <= q["tolerance"]
    # 대표 오답(σ = 2, σ/n = 0.08)은 허용 오차 밖이다
    for wrong in (sigma, sigma / n):
        assert abs(wrong - q["answer"]) > q["tolerance"]
