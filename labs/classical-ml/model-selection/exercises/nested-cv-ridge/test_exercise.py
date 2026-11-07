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


mod = load()
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def test_ridge_fit():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(30, 4)), rng.normal(size=30)
    w = mod.ridge_fit(X, y, 2.0)
    assert np.shape(w) == (4,)
    assert np.allclose(X.T @ (X @ w - y) + 2.0 * w, 0), "정규방정식 (X^T X + lam I) w = X^T y를 만족해야 합니다"
    assert np.allclose(mod.ridge_fit(X, y, 0.0), np.linalg.lstsq(X, y, rcond=None)[0]), "lam = 0이면 최소제곱해입니다"


def test_cv_mse_by_hand():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 5.0])
    # k = 2: fold 0 = 샘플 0, 2 / fold 1 = 샘플 1, 3. lam = 0이면 w = Σxy / Σx²
    w0 = (2 * 2 + 4 * 5) / (4 + 16)                     # fold 0 시험, 샘플 1, 3으로 학습
    w1 = (1 * 1 + 3 * 3) / (1 + 9)                      # fold 1 시험, 샘플 0, 2로 학습
    e0 = np.mean([(1 * w0 - 1) ** 2, (3 * w0 - 3) ** 2])
    e1 = np.mean([(2 * w1 - 2) ** 2, (4 * w1 - 5) ** 2])
    assert np.isclose(mod.cv_mse(X, y, 0.0, 2), (e0 + e1) / 2), "fold 배정은 np.arange(N) % k입니다"


def test_outer_test_fold_not_used_for_selection():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(50, 5)), rng.normal(size=50)
    _, chosen = mod.nested_cv_mse(X, y, LAMBDAS)
    y2 = y.copy()
    y2[np.arange(50) % 5 == 0] += 100.0                 # 바깥 fold 0의 시험 샘플만 망가뜨린다
    _, chosen2 = mod.nested_cv_mse(X, y2, LAMBDAS)
    assert chosen[0] == chosen2[0], "바깥 fold 0의 lambda 선택에 바깥 시험 샘플이 쓰이면 안 됩니다"
    assert len(chosen) == 5


def test_estimate_definition():
    rng = np.random.default_rng(2)
    w_true = np.array([1.0, -0.5, 0.0, 0.0])
    X = rng.normal(size=(60, 4))
    y = X @ w_true + rng.normal(0, 0.5, 60)
    est, chosen = mod.nested_cv_mse(X, y, LAMBDAS)
    outer = np.arange(60) % 5
    errs = []
    for f, lam in enumerate(chosen):
        tr, te = outer != f, outer == f
        inner = [mod.cv_mse(X[tr], y[tr], l, 5) for l in LAMBDAS]
        assert lam == LAMBDAS[int(np.argmin(inner))], "안쪽 CV에서 가장 작은 MSE의 lambda를 골라야 합니다"
        w = mod.ridge_fit(X[tr], y[tr], lam)
        errs.append(np.mean((X[te] @ w - y[te]) ** 2))
    assert np.isclose(est, np.mean(errs))


def test_flat_minimum_is_optimistic_on_noise():
    rng = np.random.default_rng(3)
    gaps = []
    for _ in range(40):
        X, y = rng.normal(size=(30, 10)), rng.normal(size=30)          # 신호 없음
        flat = min(mod.cv_mse(X, y, lam, 5) for lam in LAMBDAS)
        nested, _ = mod.nested_cv_mse(X, y, LAMBDAS)
        gaps.append(nested - flat)
    assert np.mean(gaps) > 0, "평균적으로 단순 CV의 최솟값이 nested 추정보다 작아야(낙관적이어야) 합니다"
