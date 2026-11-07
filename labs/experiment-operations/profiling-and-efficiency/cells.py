"""22.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs --group torch python labs/experiment-operations/profiling-and-efficiency/cells.py
시간 측정값은 기계와 부하에 따라 다르므로 비율의 판정(여유가 큰 기준)으로만 출력한다.
cell 3과 cell 5는 PyTorch가 필요하다(exec="python"). 출력은 이 책의 실습 환경(CPU)에서 얻은 것이다.
브라우저(Pyodide)는 단일 스레드이므로, 로컬 실행에서도 NumPy의 BLAS 스레드를 1개로 맞춘다.
부하가 큰 공유 기계에서 BLAS 스레드가 깨어나는 지연이 작은 행렬 곱의 시간을 크게 흔들기 때문이다.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# %% cell 1 — 측정: warm-up, 반복, 중앙값, 그리고 결과가 같은지
import time

import numpy as np

def bench(fn, repeat=20):
    """fn을 repeat번 실행해 각 실행 시간(초)의 배열을 돌려준다"""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.array(times)                                    # (repeat,)

_table = {}
def lookup(i):                                                # 첫 호출 때만 큰 표를 만든다(지연 초기화)
    if "sin" not in _table:
        _table["sin"] = np.sin(np.arange(2_000_000) * 1e-3)
    return _table["sin"][i]

t = bench(lambda: lookup(123))
print("첫 호출이 나머지의 중앙값보다 10배 이상 느린가:", t[0] > 10 * np.median(t[1:]))

x = np.random.default_rng(0).normal(size=100_000)            # (N,)
def loop_version():
    return sum(v * v for v in x) / len(x)
def vector_version():
    return float(np.mean(x * x))
print("결과가 같은가(상대 오차 < 1e-12):", abs(loop_version() - vector_version()) < 1e-12 * vector_version())
ratio = np.median(bench(loop_version, 5)) / np.median(bench(vector_version, 5))
print("벡터화가 10배 이상 빠른가:", ratio > 10)

# %% cell 2 — cProfile: 데이터 준비와 계산 중 어디에 시간이 가는가
import cProfile
import json
import pstats

rng = np.random.default_rng(0)
records = [json.dumps(rng.normal(size=200).round(4).tolist()) for _ in range(256)]   # 디스크의 텍스트 샘플
W = rng.normal(size=(200, 32))                                # (D, H)

def decode(rec):
    return np.array(json.loads(rec))                          # (D,)

def load_batch():
    return np.stack([decode(r) for r in records])             # (B, D): 샘플마다 Python에서 파싱

def compute(xb):
    return np.tanh(xb @ W).mean()                             # (B, D) @ (D, H)

compute(load_batch())                                         # warm-up: 첫 호출의 초기화 비용 제외
prof = cProfile.Profile()
prof.enable()
for _ in range(5):
    compute(load_batch())
prof.disable()
st = pstats.Stats(prof).stats                                 # {(파일, 줄, 함수): (원시 호출, 호출, tottime, cumtime, ...)}
calls = {fn[2]: s[1] for fn, s in st.items() if fn[2] in ("load_batch", "decode", "compute")}
cum = {fn[2]: s[3] for fn, s in st.items() if fn[2] in ("load_batch", "compute")}
print("호출 횟수:", dict(sorted(calls.items())))
print("데이터 준비가 load+compute 시간의 80% 이상인가:", cum["load_batch"] > 0.8 * (cum["load_batch"] + cum["compute"]))

# %% cell 3 — PyTorch profiler: 구간 표시와 연산별 집계
import torch
from torch.profiler import ProfilerActivity, profile, record_function

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.1)

with profile(activities=[ProfilerActivity.CPU]) as prof:     # GPU가 있으면 ProfilerActivity.CUDA를 더한다
    for step in range(5):
        with record_function("load_batch"):
            xb, yb = torch.randn(256, 64), torch.randint(0, 10, (256,))   # (B, D), (B,)
        with record_function("train_step"):
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

ev = {e.key: e for e in prof.key_averages()}
print({k: ev[k].count for k in ["load_batch", "train_step", "aten::addmm", "aten::mm", "aten::relu"]})
# 표 전체: print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# trace 파일: prof.export_chrome_trace("trace.json") → Perfetto 등에서 시간축으로 본다

# %% cell 4 — batch 크기와 처리량: 호출당 고정 비용의 분할
W1 = rng.normal(size=(4, 8))                                  # (D, H)
W2 = rng.normal(size=(8, 2))                                  # (H, C)

def forward(xb):
    return np.maximum(xb @ W1, 0) @ W2                        # (B, D) → (B, C)

data = rng.normal(size=(4096, 4))                             # (N, D)
per_sample = {}
for bs in [1, 16, 256]:
    def epoch():
        for i in range(0, 512, bs):                           # 샘플 512개를 bs씩
            forward(data[i:i + bs])
    per_sample[bs] = np.median(bench(epoch, 5)) / 512         # 샘플 하나당 초
print("batch 16이 batch 1보다 샘플당 5배 이상 빠른가:", per_sample[1] > 5 * per_sample[16])
print("batch 256이 batch 1보다 샘플당 20배 이상 빠른가:", per_sample[1] > 20 * per_sample[256])
print("결과는 batch 크기와 무관한가:", np.allclose(forward(data[:256]), np.vstack([forward(r[None]) for r in data[:256]])))

# %% cell 5 — 변화의 검증: bf16 autocast 전후의 출력 비교
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 10))
xb = torch.randn(1024, 64)                                    # (B, D)
with torch.no_grad():
    ref64 = net.double()(xb.double())                         # 기준: float64
    net.float()
    out32 = net(xb)                                           # (B, C) 기본: float32
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out16 = net(xb)                                       # (B, C) bf16 autocast

def rel_err(a, b):
    return ((a.double() - b).norm() / b.norm()).item()

print("출력 dtype:", out32.dtype, out16.dtype)
print(f"float32 대 float64 상대 오차 {rel_err(out32, ref64):.0e}, bf16 대 float64 {rel_err(out16, ref64):.0e}")
agree = (out16.argmax(dim=1) == ref64.argmax(dim=1)).double().mean().item()
print(f"예측 클래스 일치 비율: bf16 {agree:.3f}")
