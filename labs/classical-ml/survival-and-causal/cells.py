"""16.8 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/classical-ml/survival-and-causal/cells.py
"""
# %% cell 1 — 중도절단을 버리거나 고장으로 치면
import numpy as np

rng = np.random.default_rng(6)
n, shape, scale = 300, 3.0, 1000.0                             # Weibull 수명 [cycle]
life = scale * rng.weibull(shape, n)                           # 참 수명 (실험에서는 일부만 보인다)
stop = rng.uniform(400, 1200, n)                               # 셀마다 실험이 끝난 시점 [cycle]
time = np.minimum(life, stop)                                  # 관측 시간
event = life <= stop                                           # True = 고장 관측, False = 중도절단
true_median = scale * np.log(2) ** (1 / shape)
print(f"고장 관측 {event.sum()}개, 중도절단 {(~event).sum()}개")
print(f"참 중앙 수명                    {true_median:6.0f} cycle")
print(f"중도절단을 버린 중앙값          {np.median(time[event]):6.0f} cycle")
print(f"중도절단을 고장으로 친 중앙값   {np.median(time):6.0f} cycle")

# %% cell 2 — Kaplan–Meier 추정을 NumPy로
def kaplan_meier(time, event):
    t_j = np.unique(time[event])                               # 고장이 관측된 시각들, (J,)
    d_j = np.array([np.sum((time == t) & event) for t in t_j]) # 그 시각의 고장 수
    n_j = np.array([np.sum(time >= t) for t in t_j])           # 그 직전까지 살아 있는(위험 집합) 수
    return t_j, np.cumprod(1 - d_j / n_j)                      # S(t_j)

def S_at(t_j, S, t):                                           # 계단 함수 S(t)
    k = np.searchsorted(t_j, t, side="right")
    return 1.0 if k == 0 else S[k - 1]

t_j, S = kaplan_meier(time, event)
print("   t    참 S(t)   KM   중도절단 버림   중도절단=고장")
for t in (400, 600, 800, 1000):
    drop = np.mean(time[event] > t)                            # 고장 관측만으로 센 생존 비율
    as_fail = np.mean(time > t)
    print(f"{t:5d}   {np.exp(-(t / scale) ** shape):.3f}   {S_at(t_j, S, t):.3f}       {drop:.3f}          {as_fail:.3f}")
print(f"KM 중앙 수명 {t_j[np.argmax(S <= 0.5)]:.0f} cycle (참값 {true_median:.0f})")

# %% cell 3 — 수명 회귀: 중도절단을 likelihood에 넣기
from scipy.optimize import minimize

temp_c = rng.uniform(15, 45, n)                                # 운전 온도 [°C]
b_true = -0.03                                                 # 1 °C당 log 수명 변화
life2 = np.exp(np.log(1500) + b_true * (temp_c - 25)) * rng.weibull(shape, n)
stop2 = rng.uniform(300, 1500, n)
t2, e2 = np.minimum(life2, stop2), life2 <= stop2              # 관측 시간, 고장 여부

def neg_loglik(th):                                            # Weibull AFT: log λ = a + b (온도 - 25)
    a, b, log_k = th
    k, lam = np.exp(log_k), np.exp(a + b * (temp_c - 25))
    z = (t2 / lam) ** k
    log_f = np.log(k) - np.log(lam) + (k - 1) * np.log(t2 / lam) - z    # 고장: 밀도
    return -np.sum(np.where(e2, log_f, -z))                    # 중도절단: 생존확률 log S = -z

b_drop = np.polyfit(temp_c[e2] - 25, np.log(t2[e2]), 1)[0]    # 고장만으로 log 수명 회귀
b_fail = np.polyfit(temp_c - 25, np.log(t2), 1)[0]            # 중도절단도 고장으로 취급
b_mle = minimize(neg_loglik, x0=[np.log(t2.mean()), 0.0, 0.0], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000}).x[1]
print(f"중도절단 {np.mean(~e2):.0%}.  10 °C 상승 시 수명 배율 exp(10 b)")
for name, b in [("참값", b_true), ("고장만 회귀", b_drop), ("중도절단=고장 회귀", b_fail), ("중도절단 포함 Weibull MLE", b_mle)]:
    print(f"  {name:22s} b = {b:+.3f}   배율 {np.exp(10 * b):.2f}")

# %% cell 4 — 교란: 단순 비교, 회귀 보정, 성향점수 가중
m = 5000
Z = rng.normal(30, 7, m)                                       # 운전 온도 [°C] (교란 변수)
T = rng.uniform(size=m) < 1 / (1 + np.exp(-0.3 * (Z - 30)))    # 더운 현장일수록 첨가제를 더 쓴다
Y = 0.2 * Z - 1.0 * T + rng.normal(0, 0.5, m)                  # 용량 감소 [%]: 첨가제의 참 효과 -1.0

def logistic_fit(X, t, iters=25):                              # Newton 방법, X: (m, p)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (t - p))
    return 1 / (1 + np.exp(-X @ w))

naive = Y[T].mean() - Y[~T].mean()
reg = np.linalg.lstsq(np.c_[np.ones(m), T, Z], Y, rcond=None)[0][1]   # Y ~ 1 + T + Z의 T 계수
e = logistic_fit(np.c_[np.ones(m), Z], T.astype(float))         # 성향점수 P(T=1 | Z)
w1, w0 = T / e, ~T / (1 - e)                                  # 역확률 가중치
ipw = np.sum(w1 * Y) / np.sum(w1) - np.sum(w0 * Y) / np.sum(w0)
T_rct = rng.uniform(size=m) < 0.5                              # 같은 현장에서 무작위로 배정했다면
Y_rct = 0.2 * Z - 1.0 * T_rct + rng.normal(0, 0.5, m)
print(f"첨가제 사용 셀의 평균 온도 {Z[T].mean():.1f} °C, 미사용 {Z[~T].mean():.1f} °C")
for name, v in [("단순 비교", naive), ("회귀 보정 (Z 포함)", reg), ("성향점수 가중(IPW)", ipw),
                ("무작위 배정 실험의 단순 비교", Y_rct[T_rct].mean() - Y_rct[~T_rct].mean())]:
    print(f"  {name:26s} 효과 추정 {v:+.2f} %   (참값 -1.00)")

# %% cell 5 — 보정하면 안 되는 변수: 매개변수와 충돌변수
M = 2.0 - 0.8 * T + rng.normal(0, 0.3, m)                      # 저항 증가 [mΩ]: 첨가제 → M → Y
Y2 = 0.2 * Z + 1.25 * M + rng.normal(0, 0.5, m)                # 첨가제의 총효과 = 1.25 × (-0.8) = -1.0
S = Y2 < np.quantile(Y2, 0.5) + 1.5 * T                        # 선별: 첨가제 셀은 느슨한 기준으로 통과

def coef_T(*cols, y=Y2, rows=slice(None)):                     # y ~ 1 + T + cols의 T 계수
    X_ = np.c_[np.ones(m), T, *cols][rows]
    return np.linalg.lstsq(X_, y[rows], rcond=None)[0][1]

for name, v in [("Z만 보정 (올바름)", coef_T(Z)), ("Z와 매개변수 M 보정", coef_T(Z, M)),
                ("Z 보정, 선별을 통과한 셀만", coef_T(Z, rows=S))]:
    print(f"  {name:26s} 효과 추정 {v:+.2f} %   (총효과 참값 -1.00)")
