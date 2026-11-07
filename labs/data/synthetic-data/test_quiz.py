"""15.6 퀴즈(content/15-data/06-synthetic-data/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

QUIZ = Path(__file__).parents[3] / "content/15-data/06-synthetic-data/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_config_reproducibility():
    assert correct("config-reproducibility") == [0, 1, 2]


def test_grid_projection():
    g = (np.arange(4) + 0.5) / 4
    grid = np.array([(a, b) for a in g for b in g])
    assert len(np.unique(grid[:, 0])) == Q["grid-projection"]["answer"] == 4
    lhs = qmc.LatinHypercube(d=2, seed=0).random(16)
    assert len(np.unique(np.floor(lhs[:, 0] * 16))) == 16


def test_grid_curse():
    assert 10**6 == Q["grid-curse"]["answer"]


def test_lhs_property():
    for seed in range(5):
        S = qmc.LatinHypercube(d=3, seed=seed).random(20)
        for j in range(3):
            assert np.array_equal(np.sort(np.floor(S[:, j] * 20)), np.arange(20))
    assert correct("lhs-property") == [0]


def test_domain_classifier_auc():
    rng = np.random.default_rng(0)
    same = np.vstack([rng.normal(size=(200, 5)), rng.normal(size=(200, 5))])
    shifted = np.vstack([rng.normal(size=(200, 5)), rng.normal(0.8, 1.0, size=(200, 5))])
    d = np.r_[np.zeros(200), np.ones(200)]
    auc = lambda X: cross_val_score(LogisticRegression(), X, d, cv=5, scoring="roc_auc").mean()
    assert abs(auc(same) - 0.5) < 0.1                   # 같은 분포면 0.5 근처
    assert auc(shifted) > 0.85                           # 다르면 높다
    assert correct("domain-classifier-auc") == [0]


def test_sim_only_claim():
    assert correct("sim-only-claim") == [0]
