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


knn_predict = load().knn_predict


def data(seed=0, n=60):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(0, 1, n), rng.uniform(200, 400, n)])   # 단위가 다른 두 특징
    y = np.sin(6 * X[:, 0]) + 0.002 * X[:, 1]
    return X, y


def test_k1_reproduces_training_targets():
    X, y = data()
    pred = knn_predict(X, y, X, 1)
    assert np.shape(pred) == (60,), f"반환 shape은 (M,) = (60,)이어야 합니다. 현재 {np.shape(pred)}"
    assert np.allclose(pred, y), "k = 1이면 학습 점에서의 예측은 자기 자신의 출력이어야 합니다"


def test_k_equals_n_gives_mean():
    X, y = data(1)
    pred = knn_predict(X, y, X[:5] + 0.1, len(y))
    assert np.allclose(pred, y.mean()), "k = N이면 모든 예측이 학습 출력의 평균이어야 합니다"


def test_unit_change_invariance():
    X, y = data(2)
    Xt = data(3)[0][:10]
    X2, Xt2 = X.copy(), Xt.copy()
    X2[:, 1] *= 1000.0
    Xt2[:, 1] *= 1000.0
    assert np.allclose(knn_predict(X, y, Xt, 5), knn_predict(X2, y, Xt2, 5)), "표준화했다면 단위를 바꿔도 예측이 같아야 합니다"


def test_large_irrelevant_feature_does_not_dominate():
    rng = np.random.default_rng(4)
    n = 400
    X = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1000, n)])   # 둘째 특징은 무관하지만 숫자가 크다
    y = (X[:, 0] > 0.5).astype(float)
    Xt = np.column_stack([np.array([0.05, 0.95]), np.array([500.0, 500.0])])
    pred = knn_predict(X, y, Xt, 5)
    assert pred[0] < 0.5 < pred[1], "표준화 없이 거리를 재면 숫자가 큰 무관한 특징이 이웃을 정합니다"


def test_hand_example():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])     # 평균 (0.5, 1), 표준편차 (0.5, 1)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = knn_predict(X, y, np.array([[0.1, 0.4]]), 2)
    # 표준화 공간에서 시험점은 (−0.8, −0.6). 거리: (−1, −1)→y=1까지 0.45, (−1, 1)→y=3까지 1.61, (1, −1)→y=2까지 1.84
    # 표준화하지 않으면 두 번째 이웃이 y=2인 점이 되어 1.5가 나온다
    assert np.allclose(pred, [(1.0 + 3.0) / 2]), "학습 데이터의 평균·표준편차로 표준화한 뒤 거리를 재야 합니다"
