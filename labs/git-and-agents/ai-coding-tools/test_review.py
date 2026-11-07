"""4.3 실습의 review_target.py에 심어 둔 결함이 실제로 결과를 바꾸는지 확인한다(해설: review_key.md).

리뷰를 마친 뒤에 읽는다. 각 테스트는 결함 하나를 드러내고, 고친 방법과 비교한다.
"""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("review_target", HERE / "review_target.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

cycle, temp_c, cell, y = rt.make_data()
X = rt.features(cycle, temp_c, cell)


def numerical_grad(f, w, h=1e-6):
    return np.array([(f(w + e) - f(w - e)) / (2 * h) for e in np.eye(w.size) * h])


def test_defect_shape_loss_broadcasts():
    w = np.random.default_rng(0).normal(size=X.shape[1])             # w = 0이면 pred가 상수라 차이가 안 보인다
    pred = X @ w                                                      # (N,)
    assert (pred - y).shape == (len(X), len(X))                       # (N,) - (N, 1) → (N, N)
    true_mse = np.mean((pred - y[:, 0]) ** 2)
    assert not np.isclose(rt.mse_loss(w, X, y, 0.0), true_mse)


def test_defect_gradient_sign_of_l2_term():
    w = np.random.default_rng(0).normal(size=X.shape[1])
    lam = 1e-2
    f = lambda v: np.mean((X @ v - y[:, 0]) ** 2) + lam * np.sum(v ** 2)   # 의도한 손실(shape 수정)
    g_num = numerical_grad(f, w)
    rel = np.linalg.norm(rt.grad(w, X, y, lam) - g_num) / np.linalg.norm(g_num)
    assert rel > 1e-2                                                 # gradient check 실패
    fixed = 2 * X.T @ (X @ w - y[:, 0]) / len(X) + 2 * lam * w
    assert np.linalg.norm(fixed - g_num) / np.linalg.norm(g_num) < 1e-6


def test_defect_metric_is_mae_not_rmse():
    r = np.array([0.0, 0.0, 0.0, 4.0])
    assert rt.rmse(r, np.zeros((4, 1))) == 1.0                        # MAE = 1
    assert np.sqrt(np.mean(r ** 2)) == 2.0                            # RMSE = 2


def test_defect_split_leaks_cells():
    idx = np.random.default_rng(1).permutation(len(X))
    train, test = idx[:360], idx[360:]
    assert set(cell[test]) <= set(cell[train])                        # 시험 셀이 모두 학습에 나온다


def ridge(X, t, lam):
    D = X.shape[1]
    return np.linalg.solve(X.T @ X / len(X) + lam * np.eye(D), X.T @ t / len(X))


def leave_one_cell_out_rmse(lam):
    Xf = np.column_stack([np.ones_like(cycle), cycle / 60, temp_c / 35, cycle * temp_c / 2100])  # 셀 one-hot 제거
    errs = []
    for c in np.unique(cell):
        tr, te = cell != c, cell == c
        w = ridge(Xf[tr], y[tr, 0], lam)
        errs.append(Xf[te] @ w - y[te, 0])
    return np.sqrt(np.mean(np.concatenate(errs) ** 2))


def test_honest_error_is_much_larger_than_reported():
    reported = 0.0043                                                 # review_target.py가 출력한 값
    honest = min(leave_one_cell_out_rmse(lam) for lam in [0.0, 1e-4, 1e-3])
    assert honest > 5 * reported
