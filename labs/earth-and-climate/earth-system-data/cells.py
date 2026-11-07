"""27.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/earth-and-climate/earth-system-data/cells.py
"""
# %% cell 1 — 위도 가중 전구 평균
import numpy as np

lat_deg = np.arange(-87.5, 90, 5.0)                  # (36,) 격자 칸 중심 위도
lon_deg = np.arange(0, 360, 5.0)                     # (72,) 격자 칸 중심 경도
LAT = np.deg2rad(lat_deg)[:, None]                   # (36, 1)
temp_k = np.repeat(300.0 - 45.0 * np.sin(LAT) ** 2, lon_deg.size, axis=1)  # (36, 72) 극이 차가운 기온장

w = np.cos(LAT) * np.ones_like(temp_k)               # (36, 72) 면적에 비례하는 가중치
print(f"단순 평균      {temp_k.mean():.2f} K")
print(f"cos 가중 평균  {(w * temp_k).sum() / w.sum():.2f} K   (해석해 300 - 45/3 = 285 K)")

polar = np.abs(lat_deg) > 60                         # (36,) 위도 60도보다 극 쪽
print(f"극 쪽 칸 비율: 개수 {polar.mean():.3f}  면적 {(w[polar].sum() / w.sum()):.3f}"
      f"  (해석해 1 - sin 60° = {1 - np.sin(np.pi / 3):.3f})")

# %% cell 2 — 경도 관례와 결측
lon180 = np.where(lon_deg >= 180, lon_deg - 360, lon_deg)   # (72,) [-180, 180)로 변환
naive = (lon_deg >= -10) & (lon_deg <= 30)          # 0–360 좌표에 -180 기준 조건을 그대로 씀
box = (lon180 >= -10) & (lon180 <= 30)
print("경도 10°W–30°E 칸 수: 변환 없이", naive.sum(), " 변환 후", box.sum())

sst_k = temp_k.copy()                                # (36, 72) 해수면 온도처럼 육지를 결측으로
land = (lat_deg[:, None] > 0) & (lat_deg[:, None] < 70) & (lon180[None, :] > -10) & (lon180[None, :] < 140)
sst_k[land] = np.nan
ok = ~np.isnan(sst_k)
print(f"np.mean            {np.mean(sst_k)}")
print(f"결측 0으로 채움    {(w * np.nan_to_num(sst_k)).sum() / w.sum():.2f} K")
print(f"유효 칸 가중 평균  {(w * sst_k)[ok].sum() / w[ok].sum():.2f} K")

# %% cell 3 — 누적량과 단위
acc_m = np.array([0.0, 0.0004, 0.0013, 0.0013, 0.0031, 0.0052])  # (6,) 예보 시작부터 누적한 강수 [m], 1시간 간격
rate_mm_h = np.round(np.diff(acc_m) * 1000, 2)       # (5,) 각 1시간 동안의 강수 [mm/h]
print("시간별 강수 mm/h", rate_mm_h)
print("5시간 합 mm     ", round(acc_m[-1] * 1000, 2))

g0_m_s2 = 9.80665                                    # 표준 중력가속도
phi_m2_s2 = 54_915.0                                 # 500 hPa 지오퍼텐셜 [m^2/s^2]
print(f"지위고도 {phi_m2_s2 / g0_m_s2:.0f} m   기온 {253.15 - 273.15:.1f} °C")

# %% cell 4 — 기후값과 편차
rng = np.random.default_rng(0)
years = np.arange(1981, 2021)                        # (40,)
month = np.tile(np.arange(12), years.size)           # (480,) 0 = 1월
year = np.repeat(years, 12)                          # (480,)
t2m_c = (12 - 10 * np.cos(2 * np.pi * (month + 0.5) / 12)   # 계절 주기
         + 0.03 * (year - 1981)                      # 추세 0.03 °C/년
         + rng.normal(0, 0.8, year.size))            # 날씨 잡음

base = (year >= 1991) & (year <= 2020)               # 기준 기간 1991–2020
clim = np.array([t2m_c[base & (month == m)].mean() for m in range(12)])  # (12,)
anom = t2m_c - clim[month]                           # (480,)
print("1월·7월 기후값", np.round(clim[[0, 6]], 2))
print(f"표준편차: 원자료 {t2m_c.std():.2f}  편차 {anom.std():.2f}")
print("기준 기간 편차 평균 = 0:", abs(anom[base].mean()) < 1e-8)
print(f"1981–1990 편차 평균 {anom[~base].mean():+.3f} °C")
