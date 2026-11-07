"""4.3 퀴즈(content/04-git-and-agents/03-ai-coding-tools/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

QUIZ = Path(__file__).parents[3] / "content/04-git-and-agents/03-ai-coding-tools/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_broadcast_elements():
    pred, y = np.zeros((100, 1)), np.zeros(100)
    assert (pred - y).size == Q["broadcast-elements"]["answer"]


def test_gradient_check_catches_sign_error_that_still_decreases_loss():
    rng = np.random.default_rng(0)
    X, t, lam = rng.normal(size=(50, 3)), rng.normal(size=50), 0.1
    L = lambda w: np.mean((X @ w - t) ** 2) + lam * w @ w
    wrong = lambda w: 2 * X.T @ (X @ w - t) / 50 - 2 * lam * w          # L2 부호 오류
    w = np.zeros(3)
    for _ in range(200):
        w -= 0.05 * wrong(w)
    assert L(w) < L(np.zeros(3))                                        # 손실은 줄었다
    v = rng.normal(size=3)
    num = np.array([(L(v + e) - L(v - e)) / 2e-6 for e in np.eye(3) * 1e-6])
    assert np.linalg.norm(wrong(v) - num) / np.linalg.norm(num) > 1e-2  # 그래도 gradient check는 실패
    assert correct("gradient-check") == [0]


def test_selection_leakage_across_seeds():
    leaky, honest = [], []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        X, y = rng.normal(size=(100, 5000)), rng.integers(0, 2, size=100)
        Xs = SelectKBest(f_classif, k=20).fit_transform(X, y)
        leaky.append(cross_val_score(LogisticRegression(), Xs, y, cv=5).mean())
        pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
        honest.append(cross_val_score(pipe, X, y, cv=5).mean())
    assert min(leaky) > 0.7                                             # seed를 바꿔도 부풀려진다
    assert abs(np.mean(honest) - 0.5) < 0.12
    assert correct("selection-leakage") == [0]


def test_non_computational_answers():
    assert correct("agent-vs-chat") == [0]
    assert correct("ai-citation") == [0, 1]
    assert correct("task-sizing") == [0]
