"""16.6 퀴즈(content/16-classical-ml/06-model-selection/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_percentage_error

QUIZ = Path(__file__).parents[3] / "content/16-classical-ml/06-model-selection/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_kfold_vs_group():
    assert correct("kfold-vs-group") == [0]


def test_mape_near_zero():
    mape = 100 * mean_absolute_percentage_error([0.5, 10.0], [1.0, 10.5])
    assert abs(mape - Q["mape-near-zero"]["answer"]) <= Q["mape-near-zero"]["tolerance"]


def test_accuracy_imbalanced():
    # 무작위 점수의 PR-AUC는 양성 비율 근처다(선택지 2의 오개념)
    y = (rng.uniform(size=200_000) < 0.01).astype(int)
    assert abs(average_precision_score(y, rng.uniform(size=len(y))) - 0.01) < 0.003
    assert correct("accuracy-imbalanced") == [0]


def test_cost_threshold_value():
    c_fn, c_fp = 50.0, 1.0
    p = np.linspace(0, 1, 100001)
    flag = p * c_fn > (1 - p) * c_fp                         # 불량 판정이 기대 비용상 유리한 p
    thr = p[np.argmax(flag)]
    assert np.isclose(thr, c_fp / (c_fp + c_fn), atol=1e-4)
    assert abs(thr - Q["cost-threshold-value"]["answer"]) <= Q["cost-threshold-value"]["tolerance"]


def test_best_score_is_optimistic():
    # label이 무관한 데이터에서 50개 후보 중 최고 점수는 0.5보다 체계적으로 높다
    best = []
    for _ in range(100):
        y = rng.permutation(np.repeat([0, 1], 20))
        X = rng.normal(size=(40, 50))
        acc = ((X > 0) == y[:, None]).mean(0)               # 특징 부호로 예측하는 후보 50개
        best.append(np.maximum(acc, 1 - acc).max())
    assert np.mean(best) > 0.65
    assert correct("best-score-is-optimistic") == [0]


def test_nested_cv_estimates_procedure():
    assert correct("nested-cv-estimates-procedure") == [0]
