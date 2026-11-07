"""9.2 실습: 수학적으로 동등한 계산식이 다른 결과를 내는 세 사례를 만들고, 원인이 문제의 조건인지 알고리즘인지 가린다.

각 사례마다 '문제의 조건수'를 따로 계산해 두고, 두 계산식의 상대 오차를 그 옆에 놓는다.
  - 오차가 cond * eps 수준이면: 알고리즘은 할 만큼 했다. 남은 오차는 문제(조건수) 탓이다.
  - 오차가 cond * eps보다 훨씬 크면: 알고리즘이 불안정하다. 식을 바꾸면 고칠 수 있다.

실행: uv run --project labs python labs/numerical-methods/conditioning-and-stability/equivalent_formulas.py
"""
import math
from decimal import Decimal, getcontext

import numpy as np

EPS = np.finfo(float).eps
getcontext().prec = 50


def rel(a, b):
    return abs(a - b) / abs(b)


def exp_neg_series(x, terms=200):
    """e^{-x} = sum (-x)^k / k!  — 부호가 번갈아 바뀌는 급수를 그대로 더한다."""
    s, t = 0.0, 1.0
    for k in range(terms):
        s += t
        t *= -x / (k + 1)
    return s


def exp_neg_reciprocal(x, terms=200):
    """e^{-x} = 1 / sum x^k / k!  — 양수 항만 더한 뒤 역수를 취한다."""
    s, t = 0.0, 1.0
    for k in range(terms):
        s += t
        t *= x / (k + 1)
    return 1.0 / s


def case_exp():
    print("[1] e^{-x}를 Taylor 급수로: 문제의 조건수 = |x|")
    print(f"{'x':>6} {'cond':>8} {'cond*eps':>9} {'교대 급수':>10} {'1/급수':>9}")
    for x in [1.0, 5.0, 10.0, 20.0, 30.0]:
        exact = math.exp(-x)
        print(f"{x:6.0f} {x:8.0f} {x * EPS:9.1e} {rel(exp_neg_series(x), exact):10.1e} "
              f"{rel(exp_neg_reciprocal(x), exact):9.1e}")


def case_quadratic():
    print()
    print("[2] x^2 - B x + 1 = 0의 작은 근: 계수에 대한 조건수 ≈ 1")
    print(f"{'B':>6} {'cond':>8} {'cond*eps':>9} {'근의 공식':>10} {'Vieta':>9}")
    for B in [1e2, 1e4, 1e6, 1e8]:
        Bd = Decimal(B)
        exact = float((Bd - (Bd * Bd - 4).sqrt()) / 2)   # 50자리 십진 연산으로 계산한 참값
        d = math.sqrt(B * B - 4)
        naive = (B - d) / 2                      # 근의 공식 그대로
        vieta = 1 / ((B + d) / 2)                # 작은 근 = 1 / 큰 근
        print(f"{B:6.0e} {1:8.0f} {EPS:9.1e} {rel(naive, exact):10.1e} {rel(vieta, exact):9.1e}")


def case_lstsq():
    print()
    print("[3] 다항식 최소제곱(잡음 없음): 문제의 조건수 ≈ cond(X)")
    print(f"{'차수':>4} {'cond(X)':>9} {'cond*eps':>9} {'cond^2*eps':>10} {'정규방정식':>10} {'QR':>9} {'SVD':>9}")
    t = np.linspace(0.0, 1.0, 50)                # (50,)
    for degree in [4, 8, 12]:
        X = np.vander(t, degree + 1, increasing=True)      # (50, degree+1)
        w = np.ones(degree + 1)
        y = X @ w
        kappa = np.linalg.cond(X)
        w_ne = np.linalg.solve(X.T @ X, X.T @ y)
        Q, R = np.linalg.qr(X)
        w_qr = np.linalg.solve(R, Q.T @ y)
        w_svd = np.linalg.lstsq(X, y, rcond=None)[0]
        errs = [np.linalg.norm(v - w) / np.linalg.norm(w) for v in (w_ne, w_qr, w_svd)]
        print(f"{degree:4d} {kappa:9.1e} {kappa * EPS:9.1e} {kappa**2 * EPS:10.1e} "
              f"{errs[0]:10.1e} {errs[1]:9.1e} {errs[2]:9.1e}")


if __name__ == "__main__":
    case_exp()
    case_quadratic()
    case_lstsq()
