"""15.4 퀴즈(content/15-data/04-splitting-and-leakage/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

QUIZ = Path(__file__).parents[3] / "content/15-data/04-splitting-and-leakage/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_split_matches_claim():
    assert correct("split-matches-claim") == [0]


def test_window_gap():
    W, H = 30, 10
    span = lambda s: set(range(s, s + W + H))           # window 시작점 s가 쓰는 시점
    smallest = min(d for d in range(1, 200) if not (span(0) & span(d)))
    assert smallest == Q["window-gap"]["answer"] == W + H


def test_feature_selection_leak():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(100, 2000)), rng.integers(0, 2, 100)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    leaky = cross_val_score(LogisticRegression(), SelectKBest(f_classif, k=20).fit_transform(X, y), y, cv=cv).mean()
    honest = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression()), X, y, cv=cv).mean()
    assert leaky > 0.75 and honest < 0.65
    assert correct("feature-selection-leak") == [0]


def test_leakage_types():
    assert correct("leakage-types") == [0, 1, 2]


def test_best_of_many():
    rng = np.random.default_rng(2)
    news = []
    for _ in range(20):
        Xv, yv = rng.normal(size=(200, 10)), rng.integers(0, 2, 200)
        Xn, yn = rng.normal(size=(2000, 10)), rng.integers(0, 2, 2000)
        W = rng.normal(size=(500, 10))
        val_acc = (((Xv @ W.T) > 0) == yv[:, None]).mean(axis=0)     # (500,)
        best = W[np.argmax(val_acc)]
        assert val_acc.max() > 0.55                                   # 고르는 데 쓴 집합에서는 부풀려진다
        news.append(np.mean((Xn @ best > 0) == yn))
    q = Q["best-of-many"]
    assert abs(np.mean(news) - q["answer"]) <= q["tolerance"]


def test_test_reuse_and_overlap():
    assert correct("test-reuse") == [0]
    assert correct("pretraining-overlap") == [0]
