import numpy as np


def ridge_fit(X, y, lam):
    """절편 없는 ridge 회귀. X: (N, D), y: (N,). 반환: w (D,)"""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    D = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(D), X.T @ y)


def cv_mse(X, y, lam, k):
    """fold 배정 np.arange(N) % k로 k-fold CV 평균 MSE를 계산한다."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    fold = np.arange(len(y)) % k                                # (N,)
    errs = []
    for f in range(k):
        tr, te = fold != f, fold == f
        w = ridge_fit(X[tr], y[tr], lam)
        errs.append(np.mean((X[te] @ w - y[te]) ** 2))
    return float(np.mean(errs))


def nested_cv_mse(X, y, lambdas, k_outer=5, k_inner=5):
    """바깥 fold의 학습 부분 안에서만 lambda를 고르는 nested CV.

    반환: (바깥 fold MSE의 평균, 바깥 fold마다 고른 lambda 리스트)
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    outer = np.arange(len(y)) % k_outer
    errs, chosen = [], []
    for f in range(k_outer):
        tr, te = outer != f, outer == f
        inner = [cv_mse(X[tr], y[tr], lam, k_inner) for lam in lambdas]   # 학습 부분만 사용
        lam = lambdas[int(np.argmin(inner))]
        w = ridge_fit(X[tr], y[tr], lam)
        errs.append(np.mean((X[te] @ w - y[te]) ** 2))
        chosen.append(lam)
    return float(np.mean(errs)), chosen
