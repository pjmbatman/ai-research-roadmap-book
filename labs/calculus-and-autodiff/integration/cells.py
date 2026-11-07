"""8.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/calculus-and-autodiff/integration/cells.py
"""
# %% cell 1 — 삼각형 영역의 반복적분, 두 순서
import numpy as np
from scipy import integrate

f = lambda x, y: x * y**2
# dblquad(func(안쪽, 바깥), 바깥 하한, 바깥 상한, 안쪽 하한(바깥), 안쪽 상한(바깥))
I_dy_dx, _ = integrate.dblquad(lambda y, x: f(x, y), 0, 1, lambda x: 0, lambda x: x)
I_dx_dy, _ = integrate.dblquad(lambda x, y: f(x, y), 0, 1, lambda y: y, lambda y: 1)

n = 2000                                            # 격자 Riemann 합
c = (np.arange(n) + 0.5) / n                        # (n,) 칸의 중심
Xg, Yg = np.meshgrid(c, c, indexing="ij")           # (n, n)
riemann = np.sum(f(Xg, Yg) * (Yg <= Xg)) / n**2
print(f"dy 먼저 {I_dy_dx:.6f}   dx 먼저 {I_dx_dy:.6f}   Riemann {riemann:.6f}   1/15 = {1/15:.6f}")

# %% cell 2 — Jacobian 행렬식은 넓이의 배율
rng = np.random.default_rng(0)
A = np.array([[2.0, 1.0], [0.5, 1.5]])
corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) @ A.T   # (4, 2) 정사각형 꼭짓점의 상
lo, hi = corners.min(0), corners.max(0)
box = rng.uniform(lo, hi, size=(200_000, 2))        # (N, 2) 경계 상자 위 균등 표본
pre = np.linalg.solve(A, box.T).T                    # (N, 2) 역사상으로 되돌린 점
frac = np.mean(np.all((pre >= 0) & (pre <= 1), axis=1))   # 평행사변형 안의 비율
print(f"평행사변형 넓이 (MC) {frac * np.prod(hi - lo):.3f}   |det A| = {abs(np.linalg.det(A)):.3f}")

m = 400                                              # 극좌표 격자로 단위원의 넓이
r = (np.arange(m) + 0.5) / m                         # (m,)
dr, dth = 1 / m, 2 * np.pi / m
print(f"원의 넓이: r 포함 {np.sum(r) * m * dr * dth:.4f}   r 빠뜨림 {m * m * dr * dth:.4f}   pi = {np.pi:.4f}")

# %% cell 3 — 가우시안 적분과 정규화 상수
I1, _ = integrate.quad(lambda x: np.exp(-x**2 / 2), -np.inf, np.inf)
print(f"1차원: {I1:.6f}   sqrt(2 pi) = {np.sqrt(2 * np.pi):.6f}")

Sigma = np.array([[2.0, 0.8], [0.8, 1.0]])           # (2, 2) 공분산
P = np.linalg.inv(Sigma)
g = np.linspace(-10, 10, 801)
dx = g[1] - g[0]
G = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1)   # (801, 801, 2)
q = np.einsum("...i,ij,...j->...", G, P, G)               # (801, 801) 이차형식
I2 = np.exp(-0.5 * q).sum() * dx**2
print(f"2차원: {I2:.6f}   2 pi sqrt(det) = {2 * np.pi * np.sqrt(np.linalg.det(Sigma)):.6f}")

# %% cell 4 — 확률밀도의 변환: Y = exp(X)
x = rng.normal(size=1_000_000)                       # X ~ N(0, 1)
y = np.exp(x)                                        # Y = exp(X), 로그정규분포
p_x = lambda t: np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)
p_y = lambda s: p_x(np.log(s)) / s                   # |d g^{-1}/dy| = 1/y
p_wrong = lambda s: p_x(np.log(s))                   # Jacobian 항을 빠뜨림

edges = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
counts, _ = np.histogram(y, bins=edges)
for a, b, cnt in zip(edges[:-1], edges[1:], counts):
    exact, _ = integrate.quad(p_y, a, b)
    print(f"P({a:4.2f} < Y < {b:4.2f})  표본 {cnt / y.size:.4f}  공식 {exact:.4f}")
print(f"전체 적분: 올바른 밀도 {integrate.quad(p_y, 0, np.inf)[0]:.4f}"
      f"   Jacobian 누락 {integrate.quad(p_wrong, 0, np.inf)[0]:.4f}")
