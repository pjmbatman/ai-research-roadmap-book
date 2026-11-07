"""9.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/numerical-methods/nonlinear-equations/cells.py
"""
# %% cell 1 — 이분법 · secant · Newton
import numpy as np

f = lambda x: x**3 - 2 * x - 5
df = lambda x: 3 * x**2 - 2
root = 2.0945514815423265                    # 참값의 float64 값

def bisection(a, b, n):
    xs = []
    for _ in range(n):
        m = 0.5 * (a + b)
        a, b = (a, m) if f(a) * f(m) <= 0 else (m, b)   # 부호가 바뀌는 쪽을 남긴다
        xs.append(0.5 * (a + b))
    return xs

def secant(x0, x1, n):
    xs = []
    for _ in range(n):
        x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        xs.append(x1)
    return xs

def newton(x, n):
    xs = []
    for _ in range(n):
        x = x - f(x) / df(x)
        xs.append(x)
    return xs

runs = {"이분법": bisection(2, 3, 6), "secant": secant(3, 2.5, 6), "Newton": newton(3, 6)}
print("k  " + "".join(f"{name:>12s}" for name in runs))
for k in range(6):
    print(f"{k + 1}  " + "".join(f"{abs(xs[k] - root):12.1e}" for xs in runs.values()))

# %% cell 2 — 오차 수열에서 수렴 차수 추정
def order(errors):
    e = np.log(np.asarray(errors))
    return (e[2:] - e[1:-1]) / (e[1:-1] - e[:-2])      # p_k = log(e_{k+1}/e_k) / log(e_k/e_{k-1})

for name, xs in [("secant", secant(3, 2.5, 6)), ("Newton", newton(3, 5))]:
    err = [abs(x - root) for x in xs]
    print(f"{name:6s} 오차: {' '.join(f'{e:.1e}' for e in err)}")
    print(f"{'':6s} 차수 추정: {np.round(order(err), 2)}")

# %% cell 3 — 고정점 반복: 같은 근, 다른 변형
g_good = lambda x: (2 * x + 5) ** (1 / 3)     # x = (2x + 5)^(1/3)
g_bad = lambda x: (x**3 - 5) / 2              # x = (x^3 - 5) / 2
for name, g in [("(2x+5)^(1/3)", g_good), ("(x^3-5)/2", g_bad)]:
    x, xs = 2.0, []
    for _ in range(5):
        x = g(x)
        xs.append(x)
    print(f"g(x) = {name:13s} " + " ".join(f"{v:10.4g}" for v in xs))

dg = lambda g, x, h=1e-6: (g(x + h) - g(x - h)) / (2 * h)
print(f"|g'(x*)|: {abs(dg(g_good, root)):.3f} (수축)  vs  {abs(dg(g_bad, root)):.3f} (팽창)")

# %% cell 4 — 다변수 Newton: 원과 지수 곡선의 교점
from scipy.optimize import root as solve_root

def F(v):                                      # (2,) → (2,)
    x, y = v
    return np.array([x**2 + y**2 - 4, np.exp(x) + y - 1])

def J(v):                                      # (2, 2) Jacobian
    x, y = v
    return np.array([[2 * x, 2 * y],
                     [np.exp(x), 1.0]])

v = np.array([1.0, -1.0])
for k in range(6):
    print(f"k = {k}  v = {np.array2string(v, precision=10)}  |F(v)| = {np.linalg.norm(F(v)):.1e}")
    v = v - np.linalg.solve(J(v), F(v))       # J(v) dv = -F(v): 역행렬 대신 선형시스템
print("scipy.optimize.root:", solve_root(F, [1.0, -1.0], jac=J).x)

# %% cell 5 — Newton의 실패: 발산, 중근, damping
def newton_path(f, df, x, n, damped=False):
    xs = [x]
    for _ in range(n):
        step, t = f(x) / df(x), 1.0
        while damped and abs(f(x - t * step)) >= abs(f(x)) and t > 1e-4:
            t /= 2                             # backtracking: |f|가 줄 때까지 step을 반으로
        x = x - t * step
        xs.append(x)
    return xs

atan, datan = np.arctan, lambda x: 1 / (1 + x**2)
fmt = lambda xs: " ".join(f"{v:9.3g}" for v in xs)
print("arctan, x0 = 1.5        :", fmt(newton_path(atan, datan, 1.5, 5)))
print("arctan, x0 = 1.5, damped:", fmt(newton_path(atan, datan, 1.5, 5, damped=True)))

sq, dsq = lambda x: (x - 1) ** 2, lambda x: 2 * (x - 1)
err = np.abs(np.array(newton_path(sq, dsq, 2.0, 6)) - 1)
print("중근 (x-1)^2 오차 비    :", np.round(err[1:] / err[:-1], 3))
