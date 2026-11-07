"""30.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/battery-ai/battery-data/cells.py
"""
# %% cell 1 — 세 출처의 사이클 요약을 하나의 스키마로
import numpy as np
import pandas as pd

def summary(n=5, q1=3.01):                                      # 가상의 셀 하나의 사이클 요약
    k = np.arange(1, n + 1)
    return pd.DataFrame({"cycle": k, "q_dis": q1 - 0.04 * (k - 1), "i_dis": 6.0})

a = summary()                                                   # 출처 A: Ah, 방전 +, 사이클 1부터
b = summary().assign(q_dis=lambda d: d.q_dis * 1000,            # 출처 B: mAh, 방전 -, 사이클 0부터
                     i_dis=-6.0, cycle=lambda d: d.cycle - 1)
c = summary().assign(q_dis=lambda d: d.q_dis / 3.01)            # 출처 C: 첫 사이클 대비 비율만 공개
raw = pd.concat([a.assign(src="A"), b.assign(src="B"), c.assign(src="C")], ignore_index=True)
for src, d in raw.groupby("src"):
    print(f"출처 {src}: cycle {d.cycle.min()}–{d.cycle.max()}, q_dis {d.q_dis.iloc[0]:g}–{d.q_dis.iloc[-1]:g}, i_dis {d.i_dis.iloc[0]:+g}")

meta = pd.DataFrame({"unit": [1.0, 1e-3, 3.01], "sign": [1, -1, 1], "first": [1, 0, 1]},
                    index=["A", "B", "C"])                      # 출처별 변환 규칙: 문서·메타데이터에서
m = meta.loc[raw.src].reset_index(drop=True)
tidy = pd.DataFrame({"src": raw.src, "cycle": raw.cycle - m["first"] + 1,
                     "q_dis_ah": raw.q_dis * m.unit, "i_dis_a": raw.i_dis * m.sign})
tidy["soh_nom"] = tidy.q_dis_ah / 3.0                           # SOH 정의를 하나로: 공칭 3.0 Ah 기준
wide = tidy.pivot(index="cycle", columns="src", values="soh_nom")
print(wide.round(3).to_string())
print("세 출처가 일치:", np.allclose(wide.A, wide.B) and np.allclose(wide.A, wide.C),
      "/ 방전 전류가 모두 양수:", bool((tidy.i_dis_a > 0).all()))

# %% cell 2 — 사이클 단위 분할과 셀 단위 분할
from sklearn.model_selection import KFold, PredefinedSplit, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
n_cell, n_cyc = 40, 100
life = np.exp(rng.normal(np.log(900), 0.3, n_cell))             # (40,) 셀별 수명 [사이클]
q0 = rng.normal(3.05, 0.02, n_cell)                             # (40,) 초기 용량 [Ah]: 수명과 무관한 제조 편차
r0 = rng.normal(20.0, 1.0, n_cell)                              # (40,) 초기 저항 [mOhm]: 수명과 무관
k = np.arange(1, n_cyc + 1)
cap = q0[:, None] - (150 / life)[:, None] * 1e-3 * k + rng.normal(0, 0.003, (n_cell, n_cyc))
res = r0[:, None] + rng.normal(0, 0.2, (n_cell, n_cyc))         # (40, 100)
X = np.c_[cap.ravel(), (cap - cap[:, :1]).ravel(), res.ravel(), np.tile(k, n_cell)]  # (4000, 4)
y, cell = np.repeat(life, n_cyc), np.repeat(np.arange(n_cell), n_cyc)    # 사이클 한 줄 = 한 샘플
model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
pred_cyc = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
pred_cell = cross_val_predict(model, X, y, cv=PredefinedSplit(cell % 5))   # 한 셀의 사이클은 같은 fold
for name, p in [("사이클 단위 KFold", pred_cyc), ("셀 단위 5-fold", pred_cell)]:
    print(f"{name:16s} MAE {np.mean(np.abs(p - y)):6.1f} 사이클")
print(f"{'평균값 예측':16s} MAE {np.mean(np.abs(life.mean() - life)):6.1f} 사이클")

# %% cell 3 — 시험이 끝나기 전에 수명에 닿지 못한 셀
t_end = 1000                                                    # 시험 종료 사이클
observed = life <= t_end                                        # 수명이 관측된 셀
print(f"수명 관측 {observed.sum()}개, 중도절단 {(~observed).sum()}개")
print(f"전체 셀의 평균 수명(참)         {life.mean():6.0f} 사이클")
print(f"관측된 셀만의 평균 수명         {life[observed].mean():6.0f} 사이클")
print(f"중도절단을 1000으로 넣은 평균   {np.minimum(life, t_end).mean():6.0f} 사이클")

# %% cell 4 — 사이클링 용량의 잡음과 RPT 재보정
n = np.arange(1, 1301)
q_true = 3.05 - 2.2e-4 * n - 0.02 * np.exp((n - 1000) / 100)    # 표준 조건(RPT)에서의 용량 [Ah]
temp_c = 25 + 3 * np.sin(2 * np.pi * n / 7)                     # 챔버 온도의 주기적 변동
since_rpt = n % 200                                             # 200 사이클마다 RPT(긴 휴지)
q_cyc = (q_true - 0.12                                          # 고율 사이클링: RPT보다 낮게 측정
         + 0.004 * (temp_c - 25)                                # 온도가 높으면 용량이 조금 크다
         + 0.015 * np.exp(-since_rpt / 15)                      # 휴지 직후 일시적 회복
         + rng.normal(0, 0.006, n.size))                        # 측정 잡음
rpt_n = n[n % 200 == 0]
def first_below(q, thr):
    return n[np.argmax(q < thr)] if np.any(q < thr) else None
thr = 0.8 * 3.0
smooth = pd.Series(q_cyc).rolling(15, center=True, min_periods=1).median().to_numpy()
print(f"참 EOL (RPT 용량, 매 사이클)         {first_below(q_true, thr)}")
print(f"사이클링 용량, 원자료 첫 교차        {first_below(q_cyc, thr)}")
print(f"사이클링 용량, 이동 중앙값 첫 교차   {first_below(smooth, thr)}")
offset = np.median(q_true[rpt_n - 1] - smooth[rpt_n - 1])        # RPT 시점의 (RPT - 사이클링) 차이
print(f"RPT로 재보정한 사이클링 용량 교차    {first_below(smooth + offset, thr)}  (보정량 {offset * 1e3:+.0f} mAh)")
