"""10.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/optimization/optimization-basics/cells.py
"""
# %% cell 1 — feasible set 위의 최적해와 제약 없는 최적해
import numpy as np

def f(x1, x2):
    return (x1 - 2.0) ** 2 + (x2 - 1.0) ** 2

g = np.linspace(-1, 3, 401)                    # 0.01 간격 격자
X1, X2 = np.meshgrid(g, g)                     # (401, 401)
feasible = (X1 + X2 <= 1) & (X1 >= 0) & (X2 >= 0)

F = np.where(feasible, f(X1, X2), np.inf)      # 제약을 어기면 +inf
i = np.unravel_index(np.argmin(F), F.shape)
print("feasible 격자점 비율    :", round(feasible.mean(), 3))
print("제약 없는 최솟점 (2, 1) : feasible =", bool(2 + 1 <= 1), " f =", f(2, 1))
print("feasible set 위 최솟점  :", (float(X1[i]), float(X2[i])), " f =", float(F[i]))

# %% cell 2 — Hessian 고윳값으로 볼록성 판별
def hessian(fun, x, h=1e-4):
    """중앙 차분 Hessian. x: (2,) -> (2, 2)"""
    E = np.eye(2) * h
    return np.array([[(fun(x + E[i] + E[j]) - fun(x + E[i] - E[j])
                       - fun(x - E[i] + E[j]) + fun(x - E[i] - E[j])) / (4 * h * h)
                      for j in range(2)] for i in range(2)])

funcs = {
    "x^2 + 3y^2       ": lambda p: p[0] ** 2 + 3 * p[1] ** 2,
    "x^2 - y^2        ": lambda p: p[0] ** 2 - p[1] ** 2,
    "log(1 + e^(x+y)) ": lambda p: np.log1p(np.exp(p[0] + p[1])),
    "Rosenbrock       ": lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2,
}
pts = np.random.default_rng(0).uniform(-2, 2, size=(500, 2))   # (500, 2)
for name, fun in funcs.items():
    lam = np.array([np.linalg.eigvalsh(hessian(fun, p)) for p in pts])   # (500, 2)
    print(f"{name} 최소 고윳값 {lam[:, 0].min():9.3f}   PSD 비율 {np.mean(lam[:, 0] > -1e-6):.2f}")

# %% cell 3 — stationary point 분류
# f(x, y) = x^4/4 - x^2/2 + 0.2x + y^2/2,  grad = (x^3 - x + 0.2, y)
def f2(x, y):
    return x**4 / 4 - x**2 / 2 + 0.2 * x + y**2 / 2

xs = np.sort(np.roots([1, 0, -1, 0.2]).real)             # x^3 - x + 0.2 = 0의 세 근
fs = f2(xs, 0.0)
for x, fx in zip(xs, fs):
    lam = np.array([3 * x**2 - 1, 1.0])                  # Hessian = diag(3x^2 - 1, 1)
    kind = "안장점" if lam.min() < 0 else ("전역 최소" if fx == fs.min() else "국소 최소")
    print(f"x = {x:6.3f}, y = 0   f = {fx:7.4f}   고윳값 {lam.round(3)}   {kind}")

# %% cell 4 — 2계 조건으로 판정할 수 없는 경우와 제약이 있는 경우
for name, fun in [("x^3 ", lambda x: x**3), ("x^4 ", lambda x: x**4), ("-x^4", lambda x: -x**4)]:
    print(f"{name}: f'(0) = 0, f''(0) = 0,  f(-0.1) = {fun(-0.1):+.4f}, f(0.1) = {fun(0.1):+.4f}")

x_star = np.array([1.0, 0.0])                    # cell 1의 제약 최적해
grad = 2 * (x_star - np.array([2.0, 1.0]))       # (2,)
D = np.stack([X1[feasible], X2[feasible]], axis=1) - x_star   # (M, 2) feasible 방향
print("제약 최적해에서 grad =", grad, "  min grad·(x - x*) =", (D @ grad).min().round(6))
