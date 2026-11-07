"""1.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-landscape/problem-formulation/cells.py
"""
# %% cell 1 — 미래 정보가 섞인 입력
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
T = 3000
temp_c = 25.0 + np.cumsum(rng.normal(0, 0.1, size=T))     # (T,) 실내 온도의 느린 변화
obs_c = temp_c + rng.normal(0, 0.3, size=T)                # (T,) 잡음 섞인 센서 측정

def window_mean(x, lo, hi):
    """t마다 x[t+lo : t+hi+1]의 평균. 창이 범위를 벗어나는 t는 NaN."""
    out = np.full(len(x), np.nan)
    for t in range(-lo, len(x) - hi):
        out[t] = x[t + lo : t + hi + 1].mean()
    return out

target = np.roll(obs_c, -1)                                # (T,) 다음 시각의 측정값
features = {"과거 5개 평균 (t-4..t)  ": window_mean(obs_c, -4, 0),
            "중심 5개 평균 (t-2..t+2)": window_mean(obs_c, -2, 2)}
t_idx = np.arange(T)
for name, f in features.items():
    ok = ~np.isnan(f) & (t_idx < T - 1)                    # 마지막 시각은 정답이 없다
    tr, te = ok & (t_idx < 2000), ok & (t_idx >= 2000)     # 앞 2000개 학습, 뒤는 테스트
    model = LinearRegression().fit(f[tr, None], target[tr])
    mae = np.mean(np.abs(model.predict(f[te, None]) - target[te]))
    print(f"{name}  테스트 MAE {mae:.3f} °C")

# %% cell 2 — 예측 구간(horizon)에 따라 달라지는 오차
past = window_mean(obs_c, -4, 0)                           # (T,) 시각 t까지의 정보만 쓴 입력
for h in (1, 10, 50):
    y_h = np.roll(obs_c, -h)                               # (T,) h 단계 뒤의 측정값
    idx = np.arange(4, T - h)
    tr, te = idx[idx < 2000], idx[idx >= 2000]
    model = LinearRegression().fit(past[tr, None], y_h[tr])
    mae = np.mean(np.abs(model.predict(past[te, None]) - y_h[te]))
    print(f"h = {h:2d} 단계 뒤 예측  테스트 MAE {mae:.3f} °C")

# %% cell 3 — 무작위 분할과 셀 단위 분할
from sklearn.ensemble import RandomForestRegressor

n_cells, n_meas = 20, 40
cell = np.repeat(np.arange(n_cells), n_meas)               # (800,) 셀 번호
cycle = np.tile(np.linspace(0, 1000, n_meas), n_cells)     # (800,) 사이클 수
fade = rng.uniform(0.5e-4, 2.5e-4, size=n_cells)           # (20,) 셀마다 다른 열화 속도(숨은 값)
cap0_ah = rng.normal(2.0, 0.05, size=n_cells)              # (20,) 셀마다 다른 초기 용량
soh = 1.0 - fade[cell] * cycle + rng.normal(0, 0.005, size=cell.size)   # (800,)
X = np.column_stack([cycle, cap0_ah[cell]])                # (800, 2) 입력: 사이클, 초기 용량

def test_mae(is_test):
    rf = RandomForestRegressor(n_estimators=50, random_state=0)
    rf.fit(X[~is_test], soh[~is_test])
    return np.mean(np.abs(rf.predict(X[is_test]) - soh[is_test]))

random_test = rng.random(cell.size) < 0.25                 # 측정 단위로 무작위 25%
group_test = np.isin(cell, rng.choice(n_cells, 5, replace=False))   # 셀 5개를 통째로
print(f"무작위 분할   테스트 MAE {test_mae(random_test):.4f}")
print(f"셀 단위 분할  테스트 MAE {test_mae(group_test):.4f}")
