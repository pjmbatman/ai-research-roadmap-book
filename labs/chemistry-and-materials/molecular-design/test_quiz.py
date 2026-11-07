"""28.4 퀴즈(content/28-chemistry-and-materials/04-molecular-design/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/28-chemistry-and-materials/04-molecular-design/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_multiobjective():
    assert correct("multiobjective") == [0]


def test_predictor_exploitation():
    assert correct("predictor-exploitation") == [0]
    # 예측 상위 선별이 참보다 낙관적임을 확인(승자의 저주)
    rng = np.random.default_rng(0)
    true = rng.normal(size=2000)
    pred = true + rng.normal(size=2000)            # 예측 = 참 + 잡음
    top = np.argsort(pred)[::-1][:50]
    assert (pred[top] - true[top]).mean() > 0.3


def test_validity_novelty():
    assert correct("validity-novelty") == [0]


def test_optimizer_curse_fix():
    assert correct("optimizer-curse-fix") == [0, 1, 2]
    # 적용 영역 필터가 낙관 편향을 줄이는지: 잡음이 유사도와 음의 상관일 때
    rng = np.random.default_rng(1)
    true = rng.normal(size=3000)
    sim = rng.random(3000)                          # 적용 영역 지표
    noise = (1 - sim) * rng.normal(size=3000) * 2   # 적용 영역 밖일수록 큰 잡음
    pred = true + noise
    top = np.argsort(pred)[::-1][:50]
    bias_all = (pred[top] - true[top]).mean()
    idx = np.flatnonzero(sim >= np.quantile(sim, 0.7))
    top_dom = idx[np.argsort(pred[idx])[::-1][:50]]
    bias_dom = (pred[top_dom] - true[top_dom]).mean()
    assert bias_dom < bias_all
