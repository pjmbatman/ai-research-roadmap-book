"""8.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/calculus-and-autodiff/multivariable-calculus/cells.py
"""
# %% cell 1 — 편미분은 있지만 미분 가능하지 않은 함수
import numpy as np

def f(x, y):
    if x == 0 and y == 0:
        return 0.0                                  # 원점에서는 0으로 정의
    return x * y / (x**2 + y**2)

h = 1e-6
print("df/dx(0,0) ≈", (f(h, 0.0) - f(-h, 0.0)) / (2 * h))
print("df/dy(0,0) ≈", (f(0.0, h) - f(0.0, -h)) / (2 * h))
for t in [1e-1, 1e-3, 1e-6]:
    print(f"f({t:g}, {t:g}) = {f(t, t):.3f}")

# %% cell 2 — 방향미분과 gradient
def loss(p):
    x, y = p
    return x**2 + 3 * y**2

def grad(p):
    x, y = p
    return np.array([2 * x, 6 * y])                 # (2,) 열벡터

p, h = np.array([1.0, 1.0]), 1e-6
g = grad(p)
angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
U = np.stack([np.cos(angles), np.sin(angles)], axis=1)   # (3600, 2) 단위벡터
D = (np.array([loss(p + h * u) for u in U]) - loss(p)) / h  # (3600,) 방향미분
u_best = U[np.argmax(D)]
print("가장 가파른 방향 ≈", u_best.round(3), " g/|g| =", (g / np.linalg.norm(g)).round(3))
print("최대 방향미분   ≈", round(D.max(), 3), " |g| =", round(np.linalg.norm(g), 3))
tangent = np.array([-g[1], g[0]]) / np.linalg.norm(g)  # 등고선의 단위 접선
print("접선 방향의 방향미분 ≈", round((loss(p + h * tangent) - loss(p)) / h, 4))

# %% cell 3 — Hessian의 고윳값과 곡률
def rosen_hess(x, y, b=10.0):
    return np.array([[2 - 4 * b * y + 12 * b * x**2, -4 * b * x],
                     [-4 * b * x, 2 * b]])          # (2, 2)

for pt in [(1.0, 1.0), (0.0, 0.5)]:
    H = rosen_hess(*pt)
    lam, V = np.linalg.eigh(H)                      # 오름차순 고윳값, 열이 고유벡터
    print(f"점 {pt}: 고윳값 {lam.round(3)}, 최소 고윳값의 방향 {V[:, 0].round(3)}")
lam = np.linalg.eigvalsh(rosen_hess(1.0, 1.0))
print("최소점의 조건수 λmax/λmin =", round(lam[1] / lam[0], 1))

# %% cell 4 — chain rule: 층의 Jacobian 곱
rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(4, 3)), rng.normal(size=4)   # (4, 3), (4,)
W2 = rng.normal(size=(2, 4))                            # (2, 4)

def net(x):
    return W2 @ np.tanh(W1 @ x + b1)                    # (3,) -> (2,)

x = rng.normal(size=3)
a = W1 @ x + b1                                         # (4,)
J = W2 @ np.diag(1 - np.tanh(a) ** 2) @ W1              # (2,4)(4,4)(4,3) = (2, 3)

eps = 1e-6
J_fd = np.stack([(net(x + eps * e) - net(x - eps * e)) / (2 * eps)
                 for e in np.eye(3)], axis=1)           # (2, 3)
print("J shape:", J.shape)
print("chain rule와 유한차분의 최대 차이 < 1e-8:", np.abs(J - J_fd).max() < 1e-8)

# %% cell 5 — 1차·2차 Taylor 근사의 오차
def rosen(p, b=10.0):
    x, y = p
    return (1 - x) ** 2 + b * (y - x**2) ** 2

def rosen_grad(p, b=10.0):
    x, y = p
    return np.array([-2 * (1 - x) - 4 * b * x * (y - x**2), 2 * b * (y - x**2)])

a0 = np.array([0.0, 0.5])
g0, H0 = rosen_grad(a0), rosen_hess(*a0)
d = np.array([0.6, 0.8])                                # 단위 방향
for t in [0.1, 0.05, 0.025]:
    s = t * d
    e1 = abs(rosen(a0 + s) - (rosen(a0) + g0 @ s))
    e2 = abs(rosen(a0 + s) - (rosen(a0) + g0 @ s + 0.5 * s @ H0 @ s))
    print(f"|h| = {t:5.3f}  1차 오차 {e1:.2e} (/|h|^2 {e1 / t**2:.2f})  2차 오차 {e2:.2e} (/|h|^3 {e2 / t**3:.2f})")
