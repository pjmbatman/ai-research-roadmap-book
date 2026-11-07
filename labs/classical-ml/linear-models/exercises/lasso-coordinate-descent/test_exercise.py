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
soft_threshold, lasso_cd = mod.soft_threshold, mod.lasso_cd


def data(seed=0, n=80, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] + 0.3 * rng.normal(size=n)            # 상관된 두 특징
    X -= X.mean(0)
    y = X @ np.array([2.0, 0.0, -1.0, 0.0, 0.0, 0.5]) + rng.normal(0, 0.5, n)
    return X, y - y.mean()


def test_soft_threshold():
    out = soft_threshold(np.array([-3.0, -0.5, 0.0, 0.2, 2.5]), 1.0)
    assert np.allclose(out, [-2.0, 0.0, 0.0, 0.0, 1.5]), f"S(z, 1)이 틀렸습니다: {out}"
    assert np.all(out[1:4] == 0.0), "|z| ≤ t이면 정확히 0이어야 합니다"


def test_orthonormal_design():
    rng = np.random.default_rng(1)
    N = 50
    Q, _ = np.linalg.qr(rng.normal(size=(N, 4)))             # 직교 열, (N, 4)
    X = Q * np.sqrt(N)                                       # X^T X / N = I
    y = rng.normal(size=N) * 2
    w = lasso_cd(X, y, alpha=0.3)
    expected = soft_threshold(X.T @ y / N, 0.3)
    assert np.allclose(w, expected, atol=1e-8), "직교 설계에서는 w = S(X^T y / N, alpha)여야 합니다"


def test_large_alpha_gives_all_zero():
    X, y = data()
    alpha_max = np.abs(X.T @ y).max() / len(y)
    assert np.all(lasso_cd(X, y, alpha_max * 1.01) == 0.0), "alpha가 max|x_j^T y|/N 이상이면 모든 계수가 정확히 0이어야 합니다"


def test_kkt_conditions():
    X, y = data(2)
    alpha = 0.2
    w = lasso_cd(X, y, alpha)
    g = X.T @ (y - X @ w) / len(y)                           # (p,)
    nz = w != 0
    assert np.any(~nz), "이 alpha에서는 일부 계수가 정확히 0이어야 합니다"
    assert np.allclose(g[nz], alpha * np.sign(w[nz]), atol=1e-6), "0이 아닌 계수에서 x_j^T r / N = alpha·sign(w_j)여야 합니다"
    assert np.all(np.abs(g[~nz]) <= alpha + 1e-6), "0인 계수에서 |x_j^T r / N| ≤ alpha여야 합니다"
