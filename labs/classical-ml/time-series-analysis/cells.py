"""16.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/classical-ml/time-series-analysis/cells.py
"""
# %% cell 1 — 추세·계절성·자기상관 분해
import numpy as np

rng = np.random.default_rng(0)
n = 24 * 60                                                    # 60일 × 24시간
t_h = np.arange(n)                                             # 시간 [h]
ar = np.zeros(n)
for t in range(1, n):
    ar[t] = 0.8 * ar[t - 1] + rng.normal(0, 0.5)               # AR(1) 잡음
temp_c = 15 + 0.05 * t_h / 24 + 3 * np.sin(2 * np.pi * t_h / 24) + ar   # 기온 [°C], (n,)

def acf(x, lags):
    x = x - x.mean()
    return np.array([np.sum(x[:-k] * x[k:]) / np.sum(x * x) for k in lags])

coef = np.polyfit(t_h, temp_c, 1)                              # 선형 추세 [°C/h, °C]
detr = temp_c - np.polyval(coef, t_h)
season = np.array([detr[t_h % 24 == h].mean() for h in range(24)])   # (24,) 시각별 평균
resid = detr - season[t_h % 24]                                # (n,) 추세·계절성을 뺀 잔차
lags = [1, 2, 6, 12, 24]
print("시차          ", lags)
print("원 시계열 ACF ", np.round(acf(temp_c, lags), 2))
print("잔차 ACF      ", np.round(acf(resid, lags), 2))
print("AR(1) 이론값 0.8^k", np.round(0.8 ** np.array(lags), 2))
print(f"추정한 추세 {24 * coef[0]:.3f} °C/일 (참값 0.05)")

# %% cell 2 — AR(p)를 최소제곱으로 적합
def fit_ar(x, p):                                              # x_t = φ_1 x_{t-1} + ... + φ_p x_{t-p} + e_t
    L = np.column_stack([x[p - k:len(x) - k] for k in range(1, p + 1)])   # (n-p, p) lag 행렬
    phi, *_ = np.linalg.lstsq(L, x[p:], rcond=None)
    sigma = np.std(x[p:] - L @ phi)
    return phi, sigma

for p in (1, 2, 3):
    phi, sigma = fit_ar(resid, p)
    print(f"AR({p}): φ = {np.round(phi, 3)}   잔차 표준편차 {sigma:.3f}")

# %% cell 3 — 상태공간 모델: 예측·필터링·평활
q, r = 0.05**2, 0.5**2                                         # 상태 잡음, 관측 잡음 분산
x_true = np.cumsum(rng.normal(0, np.sqrt(q), 500))             # 숨은 상태: random walk, (T,)
y_obs = x_true + rng.normal(0, np.sqrt(r), 500)                # 관측, (T,)

m_pred, P_pred, m_filt, P_filt = (np.zeros(500) for _ in range(4))
m, P = 0.0, 10.0                                               # 사전 평균·분산
for t in range(500):
    m_pred[t], P_pred[t] = m, P + q                            # 예측: x_t | y_1..y_{t-1}
    K = P_pred[t] / (P_pred[t] + r)                            # Kalman gain
    m = m_pred[t] + K * (y_obs[t] - m_pred[t])                 # 필터링: x_t | y_1..y_t
    P = (1 - K) * P_pred[t]
    m_filt[t], P_filt[t] = m, P
m_smooth = m_filt.copy()                                       # 평활: x_t | y_1..y_T (RTS)
for t in range(498, -1, -1):
    G = P_filt[t] / P_pred[t + 1]
    m_smooth[t] = m_filt[t] + G * (m_smooth[t + 1] - m_pred[t + 1])

def rmse(a):
    return np.sqrt(np.mean((a[50:] - x_true[50:]) ** 2))       # 초기 50개 제외
print(f"상태 추정 RMSE  관측 그대로 {rmse(y_obs):.3f}   한 step 예측 {rmse(m_pred):.3f}"
      f"   필터링 {rmse(m_filt):.3f}   평활 {rmse(m_smooth):.3f}")

# %% cell 4 — rolling-origin backtesting과 horizon별 오차
def forecast_all(y, o, H):                                     # 시점 o까지만 보고 o+1..o+H를 예측
    th = np.arange(o + 1)
    c = np.polyfit(th, y[: o + 1], 1)
    d = y[: o + 1] - np.polyval(c, th)
    s = np.array([d[th % 24 == h].mean() for h in range(24)])
    e = d - s[th % 24]
    phi = fit_ar(e, 1)[0][0]
    fut = o + np.arange(1, H + 1)                              # 예측할 시각, (H,)
    return {"persistence": np.full(H, y[o]),
            "seasonal naive": y[fut - 24],
            "추세+계절+AR(1)": np.polyval(c, fut) + s[fut % 24] + phi ** np.arange(1, H + 1) * e[-1]}

H, origins = 24, range(24 * 40 - 1, n - 24, 24)                # 40일째부터 하루 간격, 원점 20개
err = {}
for o in origins:
    for name, f in forecast_all(temp_c, o, H).items():
        err.setdefault(name, []).append(f - temp_c[o + 1: o + 1 + H])
print(f"원점 {len(origins)}개, RMSE [°C]   h=1    h=3    h=6   h=12   h=24")
for name, e in err.items():
    rmse = np.sqrt(np.mean(np.array(e) ** 2, axis=0))          # (H,) horizon별
    print(f"{name:18s}", " ".join(f"{rmse[h - 1]:6.2f}" for h in (1, 3, 6, 12, 24)))

# %% cell 5 — 무작위 분할과 미래 정보 누수
import warnings
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore", module="threadpoolctl")      # 브라우저 실행 환경의 무해한 경고만 끈다

def lag_table(series, target, lags=(1, 2, 24)):                # 행 t: series[t-1], series[t-2], series[t-24]
    t0 = max(lags)
    F = np.column_stack([series[t0 - k: len(series) - k] for k in lags])   # (n - 24, 3)
    return F, target[t0:]

F, y_t = lag_table(temp_c, temp_c)
smooth = np.convolve(temp_c, np.ones(5) / 5, mode="same")      # 중심 이동평균: t-2..t+2를 사용
F_leak, _ = lag_table(smooth, temp_c)                          # smooth[t-1]에 y_t, y_{t+1}이 들어 있다
cvs = {"무작위 5-fold": KFold(5, shuffle=True, random_state=0), "시간 순서 5-fold": TimeSeriesSplit(5)}
for name, model, X_ in [("kNN, lag 특징", KNeighborsRegressor(5), F), ("Ridge, lag 특징", Ridge(1.0), F),
                        ("Ridge, 중심 평활한 lag", Ridge(1.0), F_leak)]:
    res = {k: np.sqrt(-cross_val_score(model, X_, y_t, cv=cv, scoring="neg_mean_squared_error").mean())
           for k, cv in cvs.items()}
    print(f"{name:16s} RMSE [°C]  " + "   ".join(f"{k} {v:.3f}" for k, v in res.items()))
