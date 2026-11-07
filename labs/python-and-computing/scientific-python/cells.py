"""5.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/python-and-computing/scientific-python/cells.py
"""
# %% cell 1 — dtype이 조용히 바꾸는 값
import numpy as np

a = np.array([1, 2, 3], dtype=np.int64)  # 정수 배열
a[0] = 2.9                               # float를 넣어도 정수로 잘린다
print("a =", a, a.dtype)

counts = np.array([200, 100], dtype=np.uint8)
print("uint8 합:", counts + counts)      # 255를 넘으면 256으로 나눈 나머지

big = np.float32(16_777_216)             # 2**24
print("float32: 2**24 + 1 == 2**24 ?", big + np.float32(1) == big)
print("평균 계산 dtype:", np.mean(np.array([1, 2], dtype=np.int64)).dtype)

# %% cell 2 — stride와 view·copy
x = np.arange(12, dtype=np.float64).reshape(3, 4)   # (3, 4)
print("shape:", x.shape, " strides(byte):", x.strides)
print("x.T strides:", x.T.strides, " C 연속인가:", x.T.flags.c_contiguous)

col = x[:, 1]                    # slicing → view: 같은 메모리를 다른 방식으로 본다
col[0] = 100.0
print("view 수정 뒤 x[0, 1] =", x[0, 1], " 메모리 공유:", np.shares_memory(col, x))

picked = x[[0, 2]]               # fancy indexing → copy
picked[:] = -1.0
mask = x > 5                     # boolean mask → copy
print("copy 수정 뒤 x[0] =", x[0], " 메모리 공유:", np.shares_memory(picked, x))
print("mask로 고른 원소 수:", x[mask].size, " shape:", x[mask].shape)

# %% cell 3 — broadcasting의 조용한 shape 확장과 벡터화
rng = np.random.default_rng(0)
N = 5
y_true = rng.normal(size=N)                 # (N,)
y_pred = y_true[:, None] + 0.1              # (N, 1) — 모델이 열벡터로 반환했다고 하자
diff = y_pred - y_true                      # (N, 1) - (N,) → (N, N)!
print("diff shape:", diff.shape)
print("잘못된 MSE:", round(float(np.mean(diff**2)), 4),
      " 올바른 MSE:", round(float(np.mean((y_pred[:, 0] - y_true)**2)), 4))

P = rng.normal(size=(200, 3))               # (N, D) 입자 좌표
def pairwise_loop(P):
    n = len(P)
    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(np.sum((P[i] - P[j]) ** 2))
    return D

def pairwise_vec(P):
    d = P[:, None, :] - P[None, :, :]       # (N, 1, D) - (1, N, D) → (N, N, D)
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d))   # (N, N)

import time
t0 = time.perf_counter(); D1 = pairwise_loop(P); t1 = time.perf_counter()
D2 = pairwise_vec(P); t2 = time.perf_counter()
print("결과 일치:", np.allclose(D1, D2), " 벡터화가 20배 이상 빠른가:", (t1 - t0) > 20 * (t2 - t1))

# %% cell 4 — pandas: groupby, merge, 시간 resampling
import pandas as pd

idx = pd.date_range("2024-07-01", periods=72, freq="h")   # 3일치 시간별 관측
hours = np.arange(72)                                    # 짝수 시각은 A, 홀수 시각은 B 관측소
obs = pd.DataFrame({
    "station": np.where(hours % 2 == 0, "A", "B"),
    "temp_c": 20 - 6 * (hours % 2) + 5 * np.sin(2 * np.pi * hours / 24),
}, index=idx)

daily = obs["temp_c"].resample("D").agg(["mean", "max"])
print(daily.round(2))

meta = pd.DataFrame({"station": ["A", "B"], "elev_m": [120, 850]})
by_station = obs.groupby("station")["temp_c"].mean().reset_index()
print(by_station.merge(meta, on="station", how="left").round(2))

# %% cell 5 — SciPy: 보간, 적분, 최적화, 희소 행렬
from scipy import integrate, interpolate, optimize, sparse

t_s = np.linspace(0, 10, 11)                       # 관측 시각
v_ms = 9.81 * t_s                                  # 자유낙하 속도
v_of_t = interpolate.CubicSpline(t_s, v_ms)
dist_m, _ = integrate.quad(v_of_t, 0, 10)          # 속도를 적분한 거리
print("보간 v(2.5 s) =", round(float(v_of_t(2.5)), 4), "m/s,  거리 =", round(dist_m, 2), "m")

res = optimize.minimize_scalar(lambda x: (x - 2.0) ** 2 + 1.0)
print("최솟값 위치:", round(res.x, 6))

n = 1000
L = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="csr")  # 1차원 Laplacian
print("희소 저장 원소 수:", L.nnz, " 밀집이면:", n * n)
