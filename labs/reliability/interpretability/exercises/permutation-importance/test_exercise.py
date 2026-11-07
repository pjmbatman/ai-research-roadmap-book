import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


permutation_importance = load().permutation_importance

rng = np.random.default_rng(0)
X = rng.normal(size=(4000, 3)) * np.array([1.0, 2.0, 1.0])   # 열마다 표준편차 1, 2, 1
w = np.array([1.5, 0.5, 0.0])                                 # 셋째 특징은 쓰지 않는다
predict = lambda A: A @ w
y = predict(X)                                                # 잡음 없는 선형 관계


def test_linear_model_theory():
    for j, var in ((0, 1.0), (1, 4.0)):
        got = permutation_importance(predict, X, y, [j])
        want = 2 * w[j] ** 2 * var
        assert abs(got - want) < 0.1 * want, f"열 {j}: 이론값 2 w² Var = {want:.2f} 근처여야 합니다. 현재 {got:.2f}"


def test_unused_feature_is_zero():
    assert abs(permutation_importance(predict, X, y, [2])) < 1e-12, "모델이 쓰지 않는 특징을 섞어도 예측은 그대로입니다"


def test_input_not_modified():
    X_copy = X.copy()
    permutation_importance(predict, X, y, [0, 1])
    assert np.array_equal(X, X_copy), "원래 X를 바꾸지 말고 복사본을 섞으세요"


def test_columns_shuffled_together():
    # x2 = x1인 데이터에서 모델이 x1 - x2를 쓰면, 함께 섞을 때는 차이가 그대로 0이라 중요도가 0이다
    a = rng.normal(size=2000)
    Xd = np.column_stack([a, a])
    pred = lambda A: A[:, 0] - A[:, 1]
    yd = np.zeros(2000)
    assert abs(permutation_importance(pred, Xd, yd, [0, 1])) < 1e-12, "cols의 열들은 같은 순열로 섞어야 합니다"
    assert permutation_importance(pred, Xd, yd, [0]) > 1.0
