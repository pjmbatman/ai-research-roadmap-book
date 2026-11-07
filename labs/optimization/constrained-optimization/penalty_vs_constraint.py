"""10.2 실습: 같은 문제를 제약식과 penalty 방식으로 풀어 해와 수렴을 비교한다.

문제: 선형회귀 손실 L(w) = ||X w - y||^2 / (2N)
- penalty(Ridge) 형:  min L(w) + (lam/2) ||w||^2            -> 닫힌 해, 그리고 경사하강
- 제약 형:            min L(w)  s.t. ||w||^2 <= r^2         -> projected gradient
- 이차 penalty 방법:  min L(w) + (rho/2) max(0, ||w||^2 - r^2)^2  -> 경사하강, rho를 키우며

r = ||w_ridge(lam)||로 두면 두 형의 해가 같고, 제약 형의 KKT 승수가 lam과 같아야 한다.

실행: uv run --project labs python labs/optimization/constrained-optimization/penalty_vs_constraint.py
"""
import numpy as np

TOL = 1e-8          # gradient mapping 노름이 이보다 작으면 수렴으로 본다
MAX_IT = 200_000


def make_data(rng, n=100, d=10):
    Z = rng.normal(size=(n, d))
    X = Z + 0.9 * Z[:, [0]]                         # 열들이 서로 상관되어 조건수가 커진다
    w_true = rng.normal(size=d)
    y = X @ w_true + 0.5 * rng.normal(size=n)
    return X, y


def ridge_closed(X, y, lam):
    n, d = X.shape
    return np.linalg.solve(X.T @ X / n + lam * np.eye(d), X.T @ y / n)


def gd(grad, w0, step, project=lambda w: w):
    """(projected) gradient descent. 반환: 해, 반복 수"""
    w = w0.copy()
    for it in range(1, MAX_IT + 1):
        w_new = project(w - step * grad(w))
        if np.linalg.norm(w_new - w) / step < TOL:
            return w_new, it
        w = w_new
    return w, MAX_IT


def main():
    rng = np.random.default_rng(0)
    X, y = make_data(rng)
    n, d = X.shape
    grad_L = lambda w: X.T @ (X @ w - y) / n
    L_smooth = np.linalg.eigvalsh(X.T @ X / n)[-1]          # 손실 gradient의 Lipschitz 상수
    w0 = np.zeros(d)
    print(f"데이터 N={n}, D={d}, X^T X / N의 조건수 {np.linalg.cond(X.T @ X / n):.1f}\n")

    print("1) Ridge(penalty) 해와 노름 제약 해의 비교")
    print("   lam    r=||w_ridge||   ||w_pen - w_con||   KKT 승수 mu   GD 반복(penalty)  PGD 반복(제약)")
    for lam in [0.01, 0.1, 1.0, 10.0]:
        w_ridge = ridge_closed(X, y, lam)
        r = np.linalg.norm(w_ridge)
        w_pen, it_pen = gd(lambda w: grad_L(w) + lam * w, w0, 1 / (L_smooth + lam))
        project = lambda w: w if np.linalg.norm(w) <= r else w * (r / np.linalg.norm(w))
        w_con, it_con = gd(grad_L, w0, 1 / L_smooth, project)
        mu = -(grad_L(w_con) @ w_con) / (w_con @ w_con)        # stationarity: grad L + mu w = 0
        print(f"  {lam:5.2f}   {r:12.4f}   {np.linalg.norm(w_pen - w_con):17.2e}   {mu:11.4f}"
              f"   {it_pen:15d}  {it_con:14d}")

    print("\n2) 이차 penalty 방법으로 같은 노름 제약을 풀 때 (lam = 1에 해당하는 r)")
    r = np.linalg.norm(ridge_closed(X, y, 1.0))
    w_star = ridge_closed(X, y, 1.0)
    print("   rho        제약 위반 ||w||^2 - r^2   ||w - w*||   GD 반복")
    for rho in [1.0, 10.0, 100.0, 1000.0]:
        def grad_P(w, rho=rho):
            excess = max(0.0, w @ w - r**2)
            return grad_L(w) + 2 * rho * excess * w
        # 이차 penalty 항의 곡률은 rho * r^2 정도로 커지므로 step도 그만큼 줄여야 안정하다
        step = 1 / (L_smooth + 8 * rho * r**2)
        w_p, it = gd(grad_P, w0, step)
        print(f"  {rho:7.0f}   {w_p @ w_p - r**2:22.3e}   {np.linalg.norm(w_p - w_star):10.2e}   {it:7d}")


if __name__ == "__main__":
    main()
