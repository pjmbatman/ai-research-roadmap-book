"""7.4 실습: 최소제곱을 정규방정식·QR·SVD(lstsq)로 풀어 정확도를 비교한다.

다항식 회귀의 설계 행렬(Vandermonde)은 차수가 높을수록 조건수가 커진다.
참 계수를 알고 잡음 없이 만든 데이터에서, 세 방법이 계수를 얼마나 정확히 되찾는지 본다.

실행: uv run --project labs python labs/linear-algebra/inner-products-and-projection/lstsq_stability.py
"""
import numpy as np


def solve_normal(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


def solve_qr(X, y):
    Q, R = np.linalg.qr(X)            # (N, D), (D, D)
    return np.linalg.solve(R, Q.T @ y)


def solve_svd(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main():
    t = np.linspace(0.0, 1.0, 50)      # (50,)
    print(f"{'차수':>4} {'cond(X)':>9} {'cond(X^T X)':>11} {'정규방정식':>11} {'QR':>9} {'lstsq':>9}")
    for degree in [3, 6, 9, 12]:
        X = np.vander(t, degree + 1, increasing=True)   # (50, degree+1)
        w_true = np.ones(degree + 1)
        y = X @ w_true                                   # 잡음 없음
        errs = []
        for solve in (solve_normal, solve_qr, solve_svd):
            w = solve(X, y)
            errs.append(np.linalg.norm(w - w_true) / np.linalg.norm(w_true))
        print(f"{degree:>4} {np.linalg.cond(X):9.1e} {np.linalg.cond(X.T @ X):11.1e} "
              f"{errs[0]:11.1e} {errs[1]:9.1e} {errs[2]:9.1e}")
    print("(오차 = |w - w_true| / |w_true|, 계수의 상대 오차)")


if __name__ == "__main__":
    main()
