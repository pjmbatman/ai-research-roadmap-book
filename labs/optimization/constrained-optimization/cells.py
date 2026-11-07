"""10.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/optimization/constrained-optimization/cells.py
"""
# %% cell 1 — 등식 제약: Lagrange 조건을 연립방정식으로 풀기
import numpy as np

# min x^2 + 2y^2  s.t.  x + y = b.   L = x^2 + 2y^2 + nu (x + y - b)
def solve(b):
    K = np.array([[2.0, 0.0, 1.0],      # dL/dx = 2x + nu = 0
                  [0.0, 4.0, 1.0],      # dL/dy = 4y + nu = 0
                  [1.0, 1.0, 0.0]])     # x + y = b
    x, y, nu = np.linalg.solve(K, [0.0, 0.0, b])
    return x, y, nu, x**2 + 2 * y**2

x, y, nu, p_star = solve(1.0)
print(f"x = {x:.4f}, y = {y:.4f}, nu = {nu:.4f}, f* = {p_star:.4f}")
print("grad f =", np.array([2 * x, 4 * y]).round(4), "  grad h = [1. 1.]")
d = 1e-3
print(f"df*/db (수치) = {(solve(1 + d)[3] - solve(1 - d)[3]) / (2 * d):.4f}   -nu = {-nu:.4f}")

# %% cell 2 — 부등식 제약: KKT 승수와 상보적 여유
# min ||x - p||^2  s.t.  a^T x <= c   (반공간으로의 투영)
p, a = np.array([2.0, 1.0]), np.array([1.0, 1.0])
for c in [0.0, 1.0, 2.0, 3.0, 4.0]:
    mu = max(0.0, 2 * (a @ p - c) / (a @ a))        # 승수: 제약이 활성일 때만 양수
    x = p - mu / 2 * a                              # stationarity: 2(x - p) + mu a = 0
    slack = c - a @ x                               # 여유: 0 이상
    print(f"c = {c:.0f}  x* = {x}  mu = {mu:.1f}  여유 = {slack:.1f}  "
          f"mu*여유 = {mu * slack:.1f}  f* = {(x - p) @ (x - p):.2f}")

# %% cell 3 — 쌍대 함수와 약·강 쌍대성
# cell 1의 문제: g(nu) = min_{x,y} L(x, y, nu) = -3 nu^2 / 8 - nu
nus = np.linspace(-4, 2, 601)
g = -3 * nus**2 / 8 - nus
i = np.argmax(g)
print(f"max g(nu) = {g[i]:.4f}  (nu = {nus[i]:.2f})   p* = {p_star:.4f}")
print("모든 nu에서 g(nu) <= p* :", bool(np.all(g <= p_star + 1e-12)))
for nu0 in [-3.0, -1.0, 1.0]:
    print(f"g({nu0:+.0f}) = {-3 * nu0**2 / 8 - nu0:+.4f}  <= p*")

# %% cell 4 — penalty 방법과 augmented Lagrangian
# cell 1의 문제를 penalty로: min x^2 + 2y^2 + (rho/2)(x + y - 1)^2 + nu (x + y - 1)
def penalized(rho, nu=0.0):
    H = np.array([[2 + rho, rho], [rho, 4 + rho]])          # (2, 2) Hessian
    x, y = np.linalg.solve(H, [rho - nu, rho - nu])
    return x + y - 1, np.linalg.cond(H)                     # 제약 위반, 조건수

for rho in [1, 10, 100, 1e4, 1e6]:
    viol, cond = penalized(rho)
    print(f"penalty  rho = {rho:>9.0f}  위반 = {viol:+.2e}  조건수 = {cond:.1e}")

nu = 0.0
for k in range(1, 6):
    viol, cond = penalized(10.0, nu)
    nu += 10.0 * viol                                       # 승수 갱신
    print(f"AL 반복 {k}  rho = 10  위반 = {viol:+.2e}  조건수 = {cond:.1e}  nu = {nu:.4f}")

# %% cell 5 — L1 근위 연산(soft thresholding)과 proximal gradient
rng = np.random.default_rng(0)
A = rng.normal(size=(50, 20))                               # (N, D)
w_true = np.zeros(20)
w_true[[2, 7, 11]] = [1.5, -2.0, 1.0]
y = A @ w_true + 0.1 * rng.normal(size=50)                  # (N,)

def soft(v, t):                                             # prox of t * ||.||_1
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

lam = 5.0
step = 1 / np.linalg.norm(A, 2) ** 2                        # 1 / L, L = sigma_max(A)^2
w = np.zeros(20)
for _ in range(500):                                        # ISTA: 0.5||Aw - y||^2 + lam ||w||_1
    w = soft(w - step * A.T @ (A @ w - y), step * lam)
w_ridge = np.linalg.solve(A.T @ A + lam * np.eye(20), A.T @ y)
print("L1 해의 0이 아닌 좌표 :", np.flatnonzero(w), " 값", w[w != 0].round(3))
print("L1 해의 정확한 0 개수 :", int(np.sum(w == 0)), "/ 20")
print("Ridge 해의 정확한 0 개수:", int(np.sum(w_ridge == 0)), "/ 20")
