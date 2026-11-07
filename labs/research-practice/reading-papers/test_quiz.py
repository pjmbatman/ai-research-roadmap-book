"""23.2 퀴즈(content/23-research-practice/02-reading-papers/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/23-research-practice/02-reading-papers/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_plus_minus_se():
    sd = 0.005 * np.sqrt(5)
    q = Q["plus-minus-se"]
    assert abs(sd - q["answer"]) <= q["tolerance"]


def test_lambda_scaling():
    rng = np.random.default_rng(0)
    N, D, lam = 500, 4, 0.01
    X = rng.normal(size=(N, D))
    y = X @ rng.normal(size=D) + rng.normal(size=N)
    w_paper = np.linalg.solve(X.T @ X / N + lam * np.eye(D), X.T @ y / N)
    lam_code = Q["lambda-scaling"]["answer"]
    w_code = np.linalg.solve(X.T @ X + lam_code * np.eye(D), X.T @ y)
    assert np.allclose(w_paper, w_code)
    assert lam_code == N * lam


def test_math_code_mismatch_scale_matters():
    # 정규화 강도가 batch 크기만큼 달라지면 해가 실제로 달라진다.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(64, 5))
    y = X @ rng.normal(size=5) + rng.normal(size=64)
    w1 = np.linalg.solve(X.T @ X + 1.0 * np.eye(5), X.T @ y)
    w2 = np.linalg.solve(X.T @ X + 64.0 * np.eye(5), X.T @ y)
    assert np.linalg.norm(w1 - w2) > 0.1
    assert correct("math-code-mismatch") == [0]


def test_other_answers():
    assert correct("claim-scope") == [0]
    assert correct("what-to-extract") == [0, 1, 2]
    assert correct("first-pass") == [0]
