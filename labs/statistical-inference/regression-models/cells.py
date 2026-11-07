"""12.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/statistical-inference/regression-models/cells.py
"""
# %% cell 1 — 선형모형: 계수와 표준오차
import numpy as np

def ols(X, y):                                   # 계수, 표준오차, 잔차 분산
    XtX_inv = np.linalg.inv(X.T @ X)             # (p, p)
    beta = XtX_inv @ X.T @ y                     # (p,)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])   # σ̂² = RSS / (n - p)
    return beta, np.sqrt(s2 * np.diag(XtX_inv)), s2

rng = np.random.default_rng(0)
cycle = np.arange(0, 201, 10.0)                  # 사이클 번호, (n,) n = 21
X = np.column_stack([np.ones_like(cycle), cycle])                 # (n, 2)
cap_ah = 2.50 - 0.0012 * cycle + rng.normal(0, 0.02, cycle.size)  # 용량 [Ah]
beta, se, _ = ols(X, cap_ah)
print(f"절편 {beta[0]:.4f} ± {se[0]:.4f} Ah,  기울기 {beta[1] * 1000:.3f} ± {se[1] * 1000:.3f} mAh/cycle")

slopes, ses = [], []
for _ in range(5000):                            # 같은 실험을 반복해 SE 공식을 확인한다
    b, s, _ = ols(X, 2.50 - 0.0012 * cycle + rng.normal(0, 0.02, cycle.size))
    slopes.append(b[1]); ses.append(s[1])
slopes, ses = np.array(slopes), np.array(ses)
cover = np.mean(np.abs(slopes + 0.0012) <= 2.093 * ses)          # t_{0.975, 19} = 2.093
print(f"기울기의 실제 sd {slopes.std() * 1000:.4f},  보고된 SE 평균 {ses.mean() * 1000:.4f} mAh/cycle,  95% 구간 포함률 {cover:.3f}")

# %% cell 2 — 일반화 선형모형: Poisson 회귀를 IRLS로 적합
def poisson_glm(X, y, iters=25):                 # log link: log E[y] = X β
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = np.exp(X @ beta)                    # (n,)
        z = X @ beta + (y - mu) / mu             # 작업 반응변수
        W = mu                                   # 가중치 = 1 / (Var(y) · (dη/dμ)²) = μ
        beta = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
    cov = np.linalg.inv(X.T @ (np.exp(X @ beta)[:, None] * X))   # Fisher 정보의 역
    return beta, np.sqrt(np.diag(cov))

temp_c = np.repeat([20.0, 30.0, 40.0, 50.0, 60.0], 8)              # 온도 [°C], (n,) n = 40
Xp = np.column_stack([np.ones_like(temp_c), (temp_c - 40) / 10])   # 10 °C 단위로 중심화
counts = rng.poisson(np.exp(1.5 + 0.35 * Xp[:, 1]))                # 시편당 미세 균열 수
b, s = poisson_glm(Xp, counts)
print(f"β1 = {b[1]:.3f} ± {s[1]:.3f}   → 10 °C마다 균열 발생률 × {np.exp(b[1]):.2f}"
      f"  (95% 구간 × {np.exp(b[1] - 1.96 * s[1]):.2f} ~ {np.exp(b[1] + 1.96 * s[1]):.2f})")
print(f"온도별 평균 균열 수: {[float(counts[temp_c == t].mean()) for t in (20, 30, 40, 50, 60)]}")

# %% cell 3 — 잔차 진단 1: 곡률과 이분산
cyc = np.arange(0, 501, 5.0)                                        # (n,) n = 101
fade = 2.5 - 0.0005 * cyc - 1.5e-6 * cyc**2                         # 가속 열화 (참 모형은 2차)
y3 = fade + rng.normal(0, 1, cyc.size) * (0.005 + 0.00006 * cyc)   # 잡음도 사이클과 함께 커진다
X3 = np.column_stack([np.ones_like(cyc), cyc])
b3, _, _ = ols(X3, y3)
fitted, r = X3 @ b3, y3 - X3 @ b3
bins = np.array_split(np.argsort(fitted), 5)                        # 적합값 순서로 5구간
print("적합값 구간      잔차 평균    잔차 sd")
for idx in bins:
    print(f"{fitted[idx].min():.3f}–{fitted[idx].max():.3f}    {r[idx].mean():+.4f}     {r[idx].std():.4f}")

# %% cell 4 — 잔차 진단 2: 자기상관과 영향점
e = np.zeros(cyc.size)
for t in range(1, cyc.size):                     # AR(1) 잡음: 이웃한 측정의 오차가 닮는다
    e[t] = 0.7 * e[t - 1] + rng.normal(0, 0.01)
r4 = (2.5 - 0.001 * cyc + e) - X3 @ ols(X3, 2.5 - 0.001 * cyc + e)[0]
dw = np.sum(np.diff(r4) ** 2) / np.sum(r4**2)
print(f"잔차의 lag-1 자기상관 {np.corrcoef(r4[:-1], r4[1:])[0, 1]:.2f},  Durbin–Watson {dw:.2f}  (독립이면 약 2)")

y5 = 2.5 - 0.001 * cyc + rng.normal(0, 0.01, cyc.size)
y5[-1] -= 0.15                                   # 마지막 사이클(가장자리 x)에 측정 오류 하나
H = X3 @ np.linalg.inv(X3.T @ X3) @ X3.T         # hat 행렬, (n, n)
h, (b5, _, s2) = np.diag(H), ols(X3, y5)
r5 = y5 - X3 @ b5
cook = r5**2 / (2 * s2) * h / (1 - h) ** 2       # Cook's distance (p = 2)
i = np.argmax(cook)
print(f"최대 Cook's distance: 사이클 {cyc[i]:.0f}, D = {cook[i]:.2f}, leverage h = {h[i]:.3f} (평균 {h.mean():.3f})")
print(f"기울기: 전체 {b5[1] * 1000:.3f}, 이 점을 빼면 {ols(X3[:-1], y5[:-1])[0][1] * 1000:.3f} mAh/cycle (참값 -1.000)")

# %% cell 5 — 반복 측정: 셀 단위 공변량의 표준오차
G, m = 12, 15                                    # 셀 12개 × 셀마다 측정 15번
hot = np.repeat(np.r_[np.zeros(6), np.ones(6)], m)                  # 고온 셀 여부 (셀 단위 공변량)
Xg = np.column_stack([np.ones(G * m), hot])
naive_cover, mean_cover = [], []
for _ in range(2000):
    u = np.repeat(rng.normal(0, 0.03, G), m)                        # 셀 고유 편차 (공유됨)
    yg = 2.45 - 0.05 * hot + u + rng.normal(0, 0.02, G * m)
    b, s, _ = ols(Xg, yg)                                            # 측정 180개를 독립으로 취급
    naive_cover.append(abs(b[1] + 0.05) <= 1.96 * s[1])
    bm, sm, _ = ols(Xg[::m], yg.reshape(G, m).mean(axis=1))          # 셀 평균 12개로 회귀
    mean_cover.append(abs(bm[1] + 0.05) <= 2.228 * sm[1])            # t_{0.975, 10}
print(f"고온 효과의 95% 구간 포함률:  측정 단위 OLS {np.mean(naive_cover):.3f}   셀 평균 회귀 {np.mean(mean_cover):.3f}")
print(f"설계 효과 1 + (m - 1)ρ = {1 + (m - 1) * 0.03**2 / (0.03**2 + 0.02**2):.1f}  (ρ = 셀 내 상관)")
