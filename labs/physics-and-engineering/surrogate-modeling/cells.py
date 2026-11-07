"""26.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/physics-and-engineering/surrogate-modeling/cells.py
"""
# %% cell 1 — 고충실도 solver로 학습 데이터 만들기
import numpy as np

def solve(nu, N=128, T=1.5):
    """u_t + (u²/2)_x = nu u_xx, 주기 경계 [0, 2π), u0 = 0.5 + sin x. 유한체적 + Heun."""
    dx = 2 * np.pi / N
    u = 0.5 + np.sin(np.arange(N) * dx)                       # (N,)
    n = int(np.ceil(T / (0.4 * min(dx / 1.5, dx**2 / (2 * nu)))))
    def rhs(u):
        ur = np.roll(u, -1)                                    # (N,) 오른쪽 이웃
        a = np.maximum(np.abs(u), np.abs(ur))
        F = 0.25 * (u**2 + ur**2) - 0.5 * a * (ur - u) - nu * (ur - u) / dx   # (N,) 면 유속
        return -(F - np.roll(F, 1)) / dx
    for _ in range(n):
        u1 = u + T / n * rhs(u)
        u = 0.5 * (u + u1 + T / n * rhs(u1))
    return u, n

nu_train = np.geomspace(0.05, 0.2, 16)                         # (16,) 학습 범위 [0.05, 0.2]
X = np.array([solve(nu)[0] for nu in nu_train])                # (16, 128) 스냅샷 행렬
print("스냅샷 행렬", X.shape, " 질량 ∫u dx 범위",
      np.round([X.sum(1).min() * 2 * np.pi / 128, X.sum(1).max() * 2 * np.pi / 128], 6))

# %% cell 2 — POD 기저와 계수 회귀
mean = X.mean(0)                                               # (128,)
_, s, Vt = np.linalg.svd(X - mean, full_matrices=False)        # Vt: (16, 128) POD 모드
print("누적 에너지 비율:", np.round(np.cumsum(s**2)[:4] / np.sum(s**2), 5))
r = 4
A = (X - mean) @ Vt[:r].T                                      # (16, r) 학습 계수
coef = [np.polyfit(np.log(nu_train), A[:, k], 3) for k in range(r)]

def surrogate(nu):
    a = np.array([np.polyval(c, np.log(nu)) for c in coef])    # (r,)
    return mean + a @ Vt[:r]                                   # (128,)

rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
for nu in [0.07, 0.13, 0.03, 0.02, 0.4]:
    u_ref = solve(nu)[0]
    proj = mean + (u_ref - mean) @ Vt[:r].T @ Vt[:r]           # 기저로 표현할 수 있는 최선
    tag = "내삽" if 0.05 <= nu <= 0.2 else "외삽"
    print(f"nu={nu:4.2f} ({tag})  대리모델 오차 {rel(surrogate(nu), u_ref):6.2%}  기저 투영 오차 {rel(proj, u_ref):6.2%}")

# %% cell 3 — 정답 없이 하는 물리 검사
mass0 = np.pi                                                  # ∫(0.5 + sin x) dx over [0, 2π)
for nu in [0.07, 0.03, 0.02, 0.4]:
    u = surrogate(nu)
    mass_err = abs(u.sum() * 2 * np.pi / 128 - mass0)
    tv = np.abs(np.diff(np.append(u, u[0]))).sum()             # 주기 총변동, 초기값 4
    bounds_ok = (u.max() <= 1.5 + 1e-9) and (u.min() >= -0.5 - 1e-9)
    print(f"nu={nu:4.2f}  질량 보존 {mass_err < 1e-10}  최대·최소 원리 {bounds_ok}  "
          f"총변동 {tv:.3f} ≤ 4 {tv <= 4}")

# %% cell 4 — 같은 정확도의 solver와 비교
u_ref = solve(0.07)[0]                                         # (128,) 기준 해
for N in [32, 64, 128]:
    u, n = solve(0.07, N=N)
    u_up = np.repeat(u, 128 // N)                              # (128,) 셀 평균을 기준 격자로
    ref_avg = np.repeat(u_ref.reshape(N, -1).mean(1), 128 // N)
    print(f"solver N={N:3d}  step {n:4d}  비용 N×step {N * n:7d}  기준 대비 오차 {rel(u_up, ref_avg):6.2%}")
print(f"대리모델     추론 비용 약 r×N = {r * 128}  기준 대비 오차 {rel(surrogate(0.07), u_ref):6.2%}  (학습 데이터 생성: solver 16회)")
