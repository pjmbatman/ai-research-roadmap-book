"""9.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/numerical-methods/conditioning-and-stability/cells.py
"""
# %% cell 1 — 조건수: 입력 섭동이 해에서 몇 배로 커지는가
import numpy as np
from scipy.linalg import hilbert

rng = np.random.default_rng(0)
for n in [5, 8, 11]:
    A = hilbert(n)                                   # (n, n) 조건이 나쁜 행렬
    x = np.ones(n)
    b = A @ x
    db = 1e-10 * rng.normal(size=n)                  # b에 작은 섭동
    dx = np.linalg.solve(A, b + db) - np.linalg.solve(A, b)
    amp = (np.linalg.norm(dx) / np.linalg.norm(x)) / (np.linalg.norm(db) / np.linalg.norm(b))
    print(f"n = {n:2d}  cond(A) = {np.linalg.cond(A):.1e}  상대 오차 증폭 = {amp:.1e}")

# %% cell 2 — forward error와 backward error: solve vs 역행렬
eps = np.finfo(float).eps
for n in [5, 8, 11]:
    A = hilbert(n)
    x = np.ones(n)
    b = A @ x
    print(f"n = {n:2d}  cond*eps = {np.linalg.cond(A) * eps:.1e}")
    for name, x_hat in [("solve", np.linalg.solve(A, b)),    # LU + 부분 피벗팅
                        ("inv @ b", np.linalg.inv(A) @ b)]:   # 역행렬을 만든 뒤 곱
        fwd = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
        bwd = np.linalg.norm(b - A @ x_hat) / (np.linalg.norm(A, 2) * np.linalg.norm(x_hat))
        print(f"   {name:8s} backward = {bwd:.1e}  forward = {fwd:.1e}")

# %% cell 3 — 조건이 좋은 문제, 불안정한 알고리즘: 이차방정식의 작은 근
b, c = -1e8, 1.0                                     # x^2 + b x + c = 0
d = np.sqrt(b * b - 4 * c)
naive = (-b - d) / 2                                 # 근의 공식 그대로: 상쇄
q = -(b - d) / 2                                     # 부호가 같은 두 수의 합
stable = c / q                                       # Vieta: x1 * x2 = c
exact = 1e-8                                         # 참값 1e-8 + 1e-24 + ...의 float64 값
for name, r in [("근의 공식", naive), ("Vieta 이용", stable)]:
    print(f"{name:8s} 작은 근 = {r:.16e}  상대 오차 = {abs(r - exact) / exact:.1e}")

# %% cell 4 — 수렴 속도: gradient descent는 조건수에 비례해 느려진다
def gd_iters(A, tol=1e-8, max_iter=100_000):
    """f(x) = x^T A x / 2를 최적 고정 step의 gradient descent로 최소화한 반복 수."""
    lam = np.linalg.eigvalsh(A)
    step = 2 / (lam[0] + lam[-1])
    x = np.ones(A.shape[0])                          # 최솟값은 x* = 0
    for k in range(max_iter):
        if np.linalg.norm(x) < tol:
            return k
        x = x - step * (A @ x)
    return max_iter

for kappa in [10, 100, 1000]:
    A = np.diag([1.0, kappa])                        # (2, 2), 조건수 kappa
    rate = (kappa - 1) / (kappa + 1)                 # 한 반복당 오차 축소율
    print(f"kappa = {kappa:5d}  반복 수 = {gd_iters(A):5d}  "
          f"예측 = {np.log(1e-8 / np.sqrt(2)) / np.log(rate):7.1f}")

# %% cell 5 — 스케일링(대각 전처리)이 조건수와 반복 수를 바꾼다
S = np.array([[2.0, 0.5],
              [0.5, 1.0]])                           # 조건이 좋은 SPD 행렬
D = np.diag([1.0, 100.0])                            # 두 변수의 단위가 100배 다름
A = D @ S @ D                                        # 같은 문제, 나쁜 스케일
d = 1 / np.sqrt(np.diag(A))
A_scaled = A * np.outer(d, d)                        # D^-1/2 A D^-1/2 (Jacobi 스케일링)
for name, M in [("원래 A", A), ("스케일 후", A_scaled)]:
    print(f"{name:6s} cond = {np.linalg.cond(M):.1e}  GD 반복 수 = {gd_iters(M)}")
