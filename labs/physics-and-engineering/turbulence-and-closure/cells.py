"""26.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/physics-and-engineering/turbulence-and-closure/cells.py
"""
# %% cell 1 — 고해상도 해를 걸러 subgrid 응력 만들기
import numpy as np

L, Nf, Nc = 2 * np.pi, 256, 32                       # 고해상도 256칸, 거친 격자 32칸
K, D = Nf // Nc, L / Nc                               # 필터 폭: 고해상도 8칸 = 거친 셀 하나

def ic(seed, N):                                      # 파수 1–5의 무작위 초기 장
    rng, x = np.random.default_rng(seed), np.arange(N) * L / N
    return sum(rng.normal() / k * np.sin(k * x + rng.uniform(0, 2 * np.pi)) for k in range(1, 6))

def rhs(u, dx, nu, tau=None):                         # u_t + (u²/2)_x = nu u_xx - tau_x, 에너지 보존형 중심 유속
    ur = np.roll(u, -1)
    F = (u**2 + u * ur + ur**2) / 6 - nu * (ur - u) / dx   # (N,) 면 유속
    if tau is not None:
        F = F + 0.5 * (tau + np.roll(tau, -1))
    return -(F - np.roll(F, 1)) / dx

def run(u, dx, nu, model=None, T=2.0, nsave=20):      # Heun 적분, 저장한 상태 (nsave, N)
    n = int(np.ceil(T / (0.3 * min(dx / 3, dx**2 / (2 * nu + 1e-12)))))
    out = []
    for i in range(n):
        u1 = u + T / n * rhs(u, dx, nu, model(u) if model else None)
        u = 0.5 * (u + u1 + T / n * rhs(u1, dx, nu, model(u1) if model else None))
        if (i + 1) % (n // nsave) == 0:
            out.append(u)
    return np.array(out)[:nsave]

filt = lambda u: u.reshape(*u.shape[:-1], -1, K).mean(-1)    # 블록 평균 필터 (..., Nf) → (..., Nc)
tau_of = lambda U: 0.5 * (filt(U**2) - filt(U) ** 2)         # subgrid 응력 τ = (평균(u²) − ū²)/2
fine = {s: run(ic(s, Nf), L / Nf, 0.04) for s in range(7)}   # 시드 0–3 학습, 4–6 시험; 각 (20, 256)
Ub, Tau = filt(fine[0]), tau_of(fine[0])                      # (20, 32), (20, 32)
print("τ의 최솟값 ≥ 0:", bool(Tau.min() >= 0), f" τ 최댓값 {Tau.max():.3f}")
print(f"거친 격자에서 τ_x 크기 / 이류항 (ū²/2)_x 크기: {np.linalg.norm(np.gradient(Tau, D, axis=1)) / np.linalg.norm(np.gradient(Ub**2 / 2, D, axis=1)):.2f}")

# %% cell 2 — a priori 평가: 걸러 낸 데이터에서 τ를 맞추는 정도
ux = lambda ub: (np.roll(ub, -1, -1) - np.roll(ub, 1, -1)) / (2 * D)
feats = lambda ub: np.stack([ux(ub)**2, -np.abs(ux(ub)) * ux(ub), ub * ux(ub), ub**2, ux(ub)], -1)
Ub_tr = np.concatenate([filt(fine[s]) for s in range(4)])     # (80, 32)
y = np.concatenate([tau_of(fine[s]) for s in range(4)]).ravel()
fit = lambda f: f.ravel() @ y / (f.ravel() @ f.ravel())        # 계수 하나의 최소제곱
smag_f = -D**2 * np.abs(ux(Ub_tr)) * ux(Ub_tr)                 # Smagorinsky: τ = −(C Δ²|ū_x|) ū_x
C_s, C_g = max(fit(smag_f), 0.0), fit(D**2 * ux(Ub_tr) ** 2)   # C_s ≥ 0으로 제한
w = np.linalg.lstsq(feats(Ub_tr).reshape(-1, 5), y, rcond=None)[0]
models = {"Smagorinsky": lambda ub: -C_s * D**2 * np.abs(ux(ub)) * ux(ub),
          "기울기 모델": lambda ub: C_g * D**2 * ux(ub) ** 2,
          "선형 회귀(5특징)": lambda ub: feats(ub) @ w}
Ub_te = np.concatenate([filt(fine[s]) for s in (4, 5, 6)])     # (60, 32)
T_te = np.concatenate([tau_of(fine[s]) for s in (4, 5, 6)])
Pi = lambda t: (t * ux(Ub_te)).sum(1) * D                    # (60,) 스냅샷별 에너지 전달 ∫ τ ū_x dx (< 0이면 소산)
print(f"참 τ         에너지 전달 평균 {Pi(T_te).mean():+.4f}  역산란(전달 > 0) 스냅샷 {np.mean(Pi(T_te) > 0):.0%}")
for name, m in models.items():
    p = m(Ub_te)
    print(f"{name:12s} 상관 {np.corrcoef(p.ravel(), T_te.ravel())[0, 1]:.3f}  "
          f"상대 오차 {np.linalg.norm(p - T_te) / np.linalg.norm(T_te):.3f}  에너지 전달 평균 {Pi(p).mean():+.4f}  역산란 스냅샷 {np.mean(Pi(p) > 0):.0%}")

# %% cell 3 — a posteriori 평가: solver에 결합해 시간 적분
def a_posteriori(nu, refs, T=2.0):
    for name, m in {"모델 없음": None, **models}.items():
        errs, e_err = [], []
        for s, U in refs.items():
            ref = filt(U)                                         # (20, 32) 걸러 낸 고해상도 궤적
            les = run(filt(ic(s, Nf)), D, nu, m, T=T)             # (20, 32) 거친 격자 + closure
            errs.append(np.linalg.norm(les[-1] - ref[-1]) / np.linalg.norm(ref[-1]))
            e_err.append(np.mean(les[-1] ** 2) / np.mean(ref[-1] ** 2) - 1)
        print(f"  {name:12s} 최종 시각 상대 오차 {np.mean(errs):.3f}  에너지 오차 {np.mean(e_err):+6.1%}")

print("nu = 0.04 (학습과 같은 조건)")
a_posteriori(0.04, {s: fine[s] for s in (4, 5, 6)})

# %% cell 4 — 새 Reynolds 수에서: 정확도와 안정성
print("nu = 0.02 (학습하지 않은 조건, 고해상도 기준 있음)")
a_posteriori(0.02, {s: run(ic(s, Nf), L / Nf, 0.02) for s in (4, 5, 6)})
print("nu = 0 (점성 없음, 기준 해 없이 물리 검사만)")
for name, m in {"모델 없음": None, **models}.items():
    u0 = filt(ic(4, Nf))
    u = run(u0, D, 0.0, m, T=4.0)[-1]
    print(f"  {name:12s} 에너지 E(4)/E(0) {np.mean(u**2) / np.mean(u0**2):.3f}  "
          f"최댓값 {np.abs(u).max():.2f} (초기 {np.abs(u0).max():.2f})")
