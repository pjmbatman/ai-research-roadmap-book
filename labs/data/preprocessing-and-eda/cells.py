"""15.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/data/preprocessing-and-eda/cells.py
"""
# %% cell 1 — 품질 점검표
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 600
t_s = np.arange(n, dtype=float)                                # (600,) 1 Hz
temp_c = 25.0 + 0.002 * t_s + rng.normal(0, 0.1, n)
temp_c[rng.choice(n, 15, replace=False)] = np.nan              # 결측
temp_c[100] = 85.0                                             # 순간 스파이크
temp_c[300:340] = 25.6                                         # 센서 멈춤: 같은 값 반복
df = pd.DataFrame({"t_s": t_s, "temp_c": temp_c})
df = pd.concat([df, df.iloc[200:205]], ignore_index=True)      # 중복 기록 5행

dup = df.duplicated()
s = df[~dup].sort_values("t_s")["temp_c"]
run_len = s.groupby(s.diff().ne(0).cumsum()).size()            # 같은 값이 이어진 길이
robust_z = 0.6745 * (s - s.median()) / (s - s.median()).abs().median()
print("결측:", int(s.isna().sum()), " 중복 행:", int(dup.sum()))
print("같은 값 최장 연속:", int(run_len.max()), "샘플")
print("robust z > 5인 시각:", s.index[robust_z.abs() > 5].tolist())

# %% cell 2 — 이상치가 표준편차를 부풀려 자신을 숨긴다
x = np.concatenate([rng.normal(10.0, 1.0, 90), rng.uniform(40.0, 60.0, 10)])  # (100,) 정상 90 + 이상치 10
z = (x - x.mean()) / x.std()
mad = np.median(np.abs(x - np.median(x)))
robust_z = 0.6745 * (x - np.median(x)) / mad
print(f"평균 {x.mean():.1f}, 표준편차 {x.std():.1f} / 중앙값 {np.median(x):.1f}, MAD {mad:.2f}")
print("|z| > 3        :", int(np.sum(np.abs(z) > 3)), "개 검출")
print("|robust z| > 3.5:", int(np.sum(np.abs(robust_z) > 3.5)), "개 검출")

# %% cell 3 — 통계량은 train에서만 추정한다
from sklearn.preprocessing import StandardScaler

t = np.arange(1000)
voltage_v = 3.7 - 0.0004 * t + rng.normal(0, 0.02, 1000)       # (1000,) 사용할수록 천천히 떨어짐
train, test = voltage_v[:800, None], voltage_v[800:, None]       # (800, 1), (200, 1) 시간 순 분할
scaler = StandardScaler().fit(train)                             # train에서만 fit
leaky = StandardScaler().fit(voltage_v[:, None])                 # test까지 보고 fit (누수)
print(f"train fit: 평균 {scaler.mean_[0]:.3f} V, 표준편차 {scaler.scale_[0]:.3f} V")
print(f"전체 fit : 평균 {leaky.mean_[0]:.3f} V, 표준편차 {leaky.scale_[0]:.3f} V")
print(f"test의 표준화 평균: train fit {scaler.transform(test).mean():.2f}, 전체 fit {leaky.transform(test).mean():.2f}")

# %% cell 4 — 재표본화와 smoothing이 지우는 것
t_irr_s = np.sort(rng.uniform(0.0, 60.0, 200))                   # (200,) 불규칙한 측정 시각
pulse = np.exp(-((t_irr_s - 30.3) / 0.8) ** 2)                   # 30.3 s에 폭 약 1 s의 펄스, 참 최댓값 1
t_grid_s = np.arange(0.0, 60.0, 0.5)                             # (120,) 2 Hz 균일 격자
on_grid = np.interp(t_grid_s, t_irr_s, pulse)                    # 선형 보간
smooth = np.convolve(on_grid, np.ones(9) / 9, mode="same")       # 4.5 s 이동 평균
coarse = np.interp(np.arange(5.0, 60.0, 10.0), t_irr_s, pulse)   # 0.1 Hz로 재표본화
for name, v in [("측정점", pulse), ("2 Hz 보간", on_grid),
                ("이동 평균", smooth), ("0.1 Hz", coarse)]:
    print(f"{name:7s} 최댓값 {v.max():.3f}")

# %% cell 5 — 전체 상관과 그룹 안 상관
parts = []
for g, (x0, y0) in enumerate([(0, 6), (2, 4), (4, 2), (6, 0)]):  # 장비마다 기준점이 다르다
    xg = x0 + rng.normal(0, 0.7, 50)
    yg = y0 + 0.8 * (xg - x0) + rng.normal(0, 0.3, 50)
    parts.append(pd.DataFrame({"device": g, "x": xg, "y": yg}))
d = pd.concat(parts, ignore_index=True)                          # (200, 3)
print("전체 상관   :", round(d["x"].corr(d["y"]), 2))
print("장비 안 상관:", " ".join(f"{g['x'].corr(g['y']):.2f}" for _, g in d.groupby("device")))
