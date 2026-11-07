"""15.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/data/storage-formats/cells.py
"""
# %% cell 1 — 같은 데이터, 세 가지 형식
import io
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 10_000
t_ms = np.arange(n, dtype=np.int64) * 100                     # (10000,) 10 Hz 타임스탬프
df = pd.DataFrame({
    "t_ms": t_ms,
    "voltage_v": 3.7 + 0.01 * np.sin(t_ms / 1000) + rng.normal(0, 1e-3, n),
    "temp_c": 25.0 + 2e-6 * t_ms + rng.normal(0, 0.05, n),
})                                                            # (10000, 3)

csv_bytes = df.to_csv(index=False).encode()
jsonl_bytes = "\n".join(json.dumps(r) for r in df.to_dict("records")).encode()
buf = io.BytesIO()
np.save(buf, df.to_numpy(dtype=np.float64))                   # (10000, 3) float64
npy_bytes = buf.getvalue()
for name, b in [("CSV", csv_bytes), ("JSONL", jsonl_bytes), ("NPY", npy_bytes)]:
    print(f"{name:5s} {len(b):>9,d} bytes")

back = np.loadtxt(io.BytesIO(csv_bytes), delimiter=",", skiprows=1)   # (10000, 3)
print("CSV 왕복 후 값이 같은가:", np.array_equal(back, df.to_numpy(dtype=np.float64)))
short = pd.read_csv(io.StringIO(df.to_csv(index=False, float_format="%.3f")))
print(f"소수 3자리 CSV의 최대 오차: {np.abs(short['voltage_v'] - df['voltage_v']).max():.1e} V")

# %% cell 2 — 열 단위로 모으면 압축이 잘 된다
import zlib

X = np.ascontiguousarray(df.to_numpy(dtype=np.float64))       # (10000, 3), 행 우선(C) 배치
row_major = X.tobytes(order="C")                              # 행 단위: t, v, T, t, v, T, ...
col_major = X.tobytes(order="F")                              # 열 단위: t, t, ..., v, v, ..., T, T, ...
t_delta = np.diff(t_ms, prepend=0).tobytes()                  # 타임스탬프의 차분: 거의 모두 100
for name, b in [("행 지향", row_major), ("열 지향", col_major)]:
    print(f"{name}: {len(b) / 1000:.1f} kB → {len(zlib.compress(b, 6)) / 1000:.1f} kB")
t_raw = len(zlib.compress(t_ms.tobytes(), 6))
print(f"t_ms 열만: 그대로 {t_raw / 1000:.1f} kB, 차분 후 {len(zlib.compress(t_delta, 6)) / 1000:.1f} kB")

# %% cell 3 — 메모리에 다 올리지 않고 덩어리(chunk)로 집계
import os
import tempfile

tmp = tempfile.mkdtemp()
csv_path = os.path.join(tmp, "series.csv")
df.to_csv(csv_path, index=False)

total, count, peak_rows = 0.0, 0, 0
for chunk in pd.read_csv(csv_path, chunksize=2_000):          # 한 번에 2000행만 메모리에
    total += chunk["temp_c"].sum()
    count += len(chunk)
    peak_rows = max(peak_rows, len(chunk))
print("행 수:", count, " 한 번에 읽은 최대 행 수:", peak_rows)
print("chunk 평균과 전체 평균의 차이 < 1e-9:", abs(total / count - df["temp_c"].mean()) < 1e-9)

# %% cell 4 — 고정 폭 이진 형식은 i번째 행으로 바로 간다
npy_path = os.path.join(tmp, "series.npy")
np.save(npy_path, X)
header = os.path.getsize(npy_path) - X.nbytes                 # NPY 헤더 크기
i, row_bytes = 7_321, X.shape[1] * 8                          # 한 행 = 3 × 8 bytes
with open(npy_path, "rb") as f:
    f.seek(header + i * row_bytes)                            # 앞의 행을 읽지 않고 건너뜀
    row = np.frombuffer(f.read(row_bytes), dtype="<f8")       # (3,)
print("헤더", header, "bytes, 읽은 행이 원본과 같은가:", np.array_equal(row, X[i]))

line_len = [len(s) for s in open(csv_path).read().splitlines()[1:]]
print("CSV 한 줄의 길이:", min(line_len), "~", max(line_len), "글자 → 위치를 계산할 수 없다")
