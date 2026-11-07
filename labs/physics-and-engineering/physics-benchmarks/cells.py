"""26.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/physics-and-engineering/physics-benchmarks/cells.py
"""
# %% cell 1 — 기준 시뮬레이터와 궤적 단위 데이터 분할
import numpy as np

N, L, DT, T = 64, 2 * np.pi, 0.02, 50              # 격자, 영역, 모델 시간 간격, rollout 길이(t = 1)
dx = L / N

def ic(seed, amp=1.0):                             # 평균 이동 + 파수 1–3의 무작위 장
    rng, x = np.random.default_rng(seed), np.arange(N) * dx
    return rng.uniform(-0.3, 0.3) + amp * sum(rng.normal() / k * np.sin(k * x + rng.uniform(0, 2 * np.pi)) for k in (1, 2, 3))

def step_ref(u, nu, sub=4):                        # 보존형 유한체적(26.1장), 모델 1 step = solver 4 step
    for _ in range(sub):
        ur = np.roll(u, -1)
        F = 0.25 * (u**2 + ur**2) - 0.5 * np.maximum(abs(u), abs(ur)) * (ur - u) - nu * (ur - u) / dx
        u = u - DT / sub / dx * (F - np.roll(F, 1))
    return u

def traj(seed, nu, amp=1.0):                       # (T+1, N) 궤적
    out = [ic(seed, amp)]
    for _ in range(T):
        out.append(step_ref(out[-1], nu))
    return np.array(out)

train = {(s, nu): traj(s, nu) for s in range(12) for nu in (0.03, 0.05, 0.1)}   # 시드 0–11 학습
X = np.concatenate([t[:-1] for t in train.values()])                         # (1800, 64) 현재 상태
Y = np.concatenate([t[1:] for t in train.values()])                          # (1800, 64) 다음 상태
NU = np.repeat([k[1] for k in train], T)[:, None]                            # (1800, 1)
print("학습 궤적", len(train), "개, 상태 쌍", X.shape, "/ 시험은 시드 100–105의 새 궤적")

# %% cell 2 — 두 학습 시뮬레이터: 직접 갱신(A)과 보존형 유속(B)
def feat_A(u, nu):                                 # 5점 스텐실의 1·2차 항 + nu × 2계 차분
    S = np.stack([np.roll(u, k, -1) for k in (2, 1, 0, -1, -2)], -1)
    quad = [S[..., i] * S[..., j] for i in range(5) for j in range(i, 5)]
    lap = np.roll(u, -1, -1) - 2 * u + np.roll(u, 1, -1)
    return np.concatenate([S, np.stack(quad, -1), (nu * lap)[..., None]], -1)   # (..., N, 21)

def flux_B(u, nu):                                 # 면 유속 = 이류 + nu 확산 + 수치 점성 꼴의 세 특징
    ur = np.roll(u, -1, -1)
    return np.stack([(u**2 + u * ur + ur**2) / 6, -nu * (ur - u), -abs(u + ur) * (ur - u)], -1)

w_A = np.linalg.lstsq(feat_A(X, NU).reshape(-1, 21), (Y - X).ravel(), rcond=None)[0]
dF = flux_B(X, NU) - np.roll(flux_B(X, NU), 1, -2)   # (1800, 64, 3) 면 유속의 차 F_{j+1/2} − F_{j−1/2}
w_B = np.linalg.lstsq(dF.reshape(-1, 3), (X - Y).ravel(), rcond=None)[0]

def step_B(u, nu):
    F = flux_B(u, nu) @ w_B                        # (N,) 면 유속: 한 셀이 잃은 양을 이웃이 얻는다
    return u - (F - np.roll(F, 1))

models = {"A 직접 갱신(21)": lambda u, nu: u + feat_A(u, nu) @ w_A, "B 보존형 유속(3)": step_B}
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
ref = traj(100, 0.07)                              # 학습하지 않은 궤적, 학습 범위 안의 nu
for name, f in models.items():
    e1 = np.mean([rel(f(ref[i], 0.07), ref[i + 1]) for i in range(T)])
    print(f"{name}  한 step 상대 오차 {e1:.4f}")

# %% cell 3 — 한 step 오차 너머: rollout, 보존량, 스펙트럼
def hi_k(u):                                       # 파수 8 이상이 담은 에너지 비율
    E = np.abs(np.fft.rfft(u)) ** 2
    return E[8:].sum() / E[1:].sum()

def evaluate(nu, amp=1.0, seeds=range(100, 106)):
    for name, f in models.items():
        e1, eT, mass, hk, blown = [], [], [], [], 0
        for s in seeds:
            ref = traj(s, nu, amp)
            e1.append(np.mean([rel(f(ref[i], nu), ref[i + 1]) for i in range(T)]))
            u = ref[0]
            for _ in range(T):
                u = f(u, nu)
                if not np.all(np.isfinite(u)) or abs(u).max() > 1e3:
                    blown += 1
                    break
            else:
                eT.append(rel(u, ref[-1]))
                mass.append(abs(u.sum() - ref[0].sum()) * dx)
                hk.append(hi_k(u) / hi_k(ref[-1]))
        m = "< 1e-12" if max(mass, default=0) < 1e-12 else f"{max(mass):7.1e}"
        tail = (f"rollout {np.mean(eT):6.3f}  질량 변화 최대 {m}  고파수 에너지 비 {np.mean(hk):5.2f}"
                if eT else "rollout     -")
        print(f"  {name}  한 step {np.mean(e1):.4f}  발산 {blown}/{len(seeds)}  {tail}")

print("학습 범위 안 (nu = 0.07)")
evaluate(0.07)

# %% cell 4 — OOD 축과 비용
for label, nu, amp in [("nu = 0.015 (학습 범위 아래)", 0.015, 1.0), ("nu = 0.3 (학습 범위 위)", 0.3, 1.0),
                       ("진폭 2배 (nu = 0.07)", 0.07, 2.0)]:
    print(label)
    evaluate(nu, amp)
print(f"파라미터 수: A {w_A.size}, B {w_B.size}.  B의 확산 계수 × nu = {w_B[1] * 0.3:.2f} (nu = 0.3, explicit 안정 한계 0.5)")
