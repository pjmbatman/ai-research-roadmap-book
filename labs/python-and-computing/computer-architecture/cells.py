"""5.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/python-and-computing/computer-architecture/cells.py
시간 측정값은 기계마다 다르므로 대소 관계처럼 안정적인 형태로만 출력한다.
cell 3–4는 스레드를 쓰므로 브라우저(Pyodide)가 아닌 로컬 Python에서 실행한다(exec="python").
"""
# %% cell 1 — 메모리 접근 순서: 연속 접근과 무작위 접근
import time
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=8_000_000)                 # 64 MB: CPU 캐시보다 크다
seq_idx = np.arange(x.size)                    # 0, 1, 2, ... 순서대로
rand_idx = rng.permutation(x.size)             # 같은 원소를 무작위 순서로

def best_time(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best

t_seq = best_time(lambda: x[seq_idx].sum())
t_rand = best_time(lambda: x[rand_idx].sum())
print("같은 합인가:", np.isclose(x[seq_idx].sum(), x[rand_idx].sum()))
print("무작위 접근이 1.5배 이상 느린가:", t_rand > 1.5 * t_seq)

A = np.zeros((1000, 1000))                     # C order (row-major)
print("C order strides:", A.strides, " Fortran order strides:", np.asfortranarray(A).strides)

# %% cell 2 — 연산 강도와 roofline
def intensity(flops, bytes_moved):
    return flops / bytes_moved                 # FLOP/byte

n, m = 10**7, 4096                             # 벡터 길이, 정사방행렬 크기 (float64 = 8 byte)
ops = {
    "axpy y=2a+b": (2 * n, 3 * 8 * n),         # 곱·합 각 1번 / a, b 읽고 y 쓰기
    "matmul C=AB": (2 * m**3, 3 * 8 * m * m),  # 최소 이동: 세 행렬을 한 번씩
}
peak_flops, bandwidth = 10e12, 1e12            # 가상의 가속기: 10 TFLOP/s, 1 TB/s
ridge = peak_flops / bandwidth                 # 이보다 연산 강도가 낮으면 메모리가 병목
for name, (flops, nbytes) in ops.items():
    I = intensity(flops, nbytes)
    attainable = min(peak_flops, I * bandwidth)          # roofline
    bound = "메모리" if I < ridge else "연산"
    print(f"{name}: 강도 {I:7.3f} FLOP/byte → 최대 {attainable / 1e12:6.3f} TFLOP/s ({bound} 한계)")

# %% cell 3 — GIL: CPU 일은 스레드로 빨라지지 않고, 기다리는 일은 빨라진다
import sys
import time
from concurrent.futures import ThreadPoolExecutor

def cpu_task(_=None):               # 순수 Python 계산: 실행 중 GIL을 쥔다
    s = 0
    for i in range(3_000_000):
        s += i * i
    return s

def io_task(_=None):                # 파일·네트워크 대기를 흉내: 기다리는 동안 GIL을 놓는다
    time.sleep(0.2)

def wall(fn, k, workers):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fn, range(k)))
    return time.perf_counter() - t0

gil = getattr(sys, "_is_gil_enabled", lambda: True)()
print("GIL 켜짐:", gil)
print("CPU 일 4개: 스레드 4개로 2배 이상 빨라지는가:", wall(cpu_task, 4, 1) > 2 * wall(cpu_task, 4, 4))
print("I/O 일 4개: 스레드 4개로 2배 이상 빨라지는가:", wall(io_task, 4, 1) > 2 * wall(io_task, 4, 4))

# %% cell 4 — 비동기 실행을 잘못 재는 법과 바르게 재는 법(GPU 커널 실행을 흉내)
import threading

def launch_kernel(duration_s=0.3):
    """GPU 커널 호출처럼, 일을 맡겨 두고 바로 돌아온다."""
    worker = threading.Thread(target=time.sleep, args=(duration_s,))
    worker.start()
    return worker

t0 = time.perf_counter()
job = launch_kernel()
t_launch = time.perf_counter() - t0       # 동기화 없이 잰 시간
job.join()                                # torch.cuda.synchronize()에 해당
t_sync = time.perf_counter() - t0
print(f"동기화 없이 잰 시간: {t_launch:.1f} s,  동기화 후 잰 시간: {t_sync:.1f} s")

# %% cell 5 — 수의 표현: 부동소수점과 고정 폭 정수
print("0.1 + 0.2 == 0.3 :", 0.1 + 0.2 == 0.3, "  isclose:", np.isclose(0.1 + 0.2, 0.3))
print(f"0.1 + 0.2 = {0.1 + 0.2:.17f}")
for dt in (np.float16, np.float32, np.float64):
    fi = np.finfo(dt)
    print(f"{dt.__name__:8s} 비트 {fi.bits:2d}  machine eps {float(fi.eps):.2e}  최댓값 {float(fi.max):.2e}")
with np.errstate(over="ignore"):
    print("float16(70000) =", np.float16(70000.0))
counts = np.array([2_000_000_000, 2_000_000_000], dtype=np.int32)
print("int32로 누적한 합:", np.add.reduce(counts, dtype=np.int32), "  int64 합:", counts.sum(dtype=np.int64))
