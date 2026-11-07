"""15.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/data/data-semantics/cells.py
"""
# %% cell 1 — 행 수와 독립 단위의 수
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n_cells, n_cycles = 8, 30
cell_effect_ah = rng.normal(0.0, 0.05, size=n_cells)        # (8,) 셀마다 다른 제조 편차
df = pd.DataFrame({
    "cell_id": np.repeat([f"C{i:02d}" for i in range(n_cells)], n_cycles),
    "cycle": np.tile(np.arange(1, n_cycles + 1), n_cells),
})                                                           # (240, 2)
df["capacity_ah"] = (2.0 + np.repeat(cell_effect_ah, n_cycles)
                     - 0.001 * df["cycle"] + rng.normal(0.0, 0.005, size=len(df)))

print("행 수:", len(df), " 셀 수:", df["cell_id"].nunique())
naive_se = df["capacity_ah"].std() / np.sqrt(len(df))
cell_means = df.groupby("cell_id")["capacity_ah"].mean()     # (8,)
cell_se = cell_means.std() / np.sqrt(len(cell_means))
print(f"행을 독립으로 본 평균의 표준오차 = {naive_se:.4f} Ah")
print(f"셀을 단위로 본 평균의 표준오차   = {cell_se:.4f} Ah")

# %% cell 2 — 계산된 label: 잡음은 줄고 편향은 남는다
t_s = np.arange(0.0, 3600.0, 1.0)                  # (3600,) 1 Hz로 1시간
current_a = np.full_like(t_s, 2.0)                 # 실제 방전 전류 2 A
for offset_a in [0.0, 0.02]:
    measured_a = current_a + offset_a + rng.normal(0.0, 0.05, size=t_s.shape)
    capacity_ah = measured_a.sum() * 1.0 / 3600.0  # 전류 적분(dt = 1 s) → Ah
    print(f"센서 offset {offset_a:.2f} A → 계산된 용량 {capacity_ah:.4f} Ah (참값 2.0000 Ah)")

# %% cell 3 — 결측 코드(sentinel)를 숫자로 믿을 때
raw = pd.DataFrame({
    "temp_c":    [25.1, 24.8, -999.0, 25.3, 25.0, -999.0],
    "voltage_v": [3.71, 3.69, 3.70, np.nan, 3.68, 3.72],
})
print("sentinel 그대로의 temp_c 평균 :", round(raw["temp_c"].mean(), 2))
clean = raw.replace({"temp_c": {-999.0: np.nan}})
print("NaN으로 바꾼 뒤의 temp_c 평균:", round(clean["temp_c"].mean(), 2))
print("열별 결측 수:", clean.isna().sum().to_dict())

# %% cell 4 — 데이터 사전으로 단위 혼합을 잡는다
dictionary = {
    "temp_c":    {"unit": "°C", "min": -20.0, "max": 60.0},
    "voltage_v": {"unit": "V",  "min": 2.5,   "max": 4.25},
}
site_a = pd.DataFrame({"temp_c": [24.9, 25.2, 26.0], "voltage_v": [3.70, 3.65, 3.61]})
site_b = pd.DataFrame({"temp_c": [298.4, 298.9, 299.3], "voltage_v": [3702, 3651, 3608]})  # 실제로는 K, mV
merged = pd.concat([site_a.assign(site="A"), site_b.assign(site="B")], ignore_index=True)

def validate(df, dictionary):
    for col, spec in dictionary.items():
        bad = df[col].notna() & ~df[col].between(spec["min"], spec["max"])
        sites = sorted(df.loc[bad, "site"].unique())
        print(f"{col:9s} [{spec['unit']}] 범위 밖 {bad.sum()}개, 출처 {sites}")

validate(merged, dictionary)
