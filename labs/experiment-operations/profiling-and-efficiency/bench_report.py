"""22.4 실습: 본문 셀의 측정을 실제 시간(ms·µs)으로 보고한다. 숫자는 기계와 부하에 따라 다르다.

실행: uv run --project labs --group torch python labs/experiment-operations/profiling-and-efficiency/bench_report.py
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import platform
import time

import numpy as np
import torch


def median_time(fn, warmup=2, repeat=15):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    print(f"환경: {platform.processor() or platform.machine()}, Python {platform.python_version()}, "
          f"NumPy {np.__version__}, PyTorch {torch.__version__}, 스레드 1개")

    x = rng.normal(size=100_000)
    t_loop = median_time(lambda: sum(v * v for v in x) / len(x), repeat=5)
    t_vec = median_time(lambda: float(np.mean(x * x)))
    print(f"[측정] 10만 원소 제곱 평균: Python 반복문 {t_loop * 1e3:.2f} ms, NumPy {t_vec * 1e3:.3f} ms "
          f"({t_loop / t_vec:.0f}배)")

    records = [json.dumps(rng.normal(size=200).round(4).tolist()) for _ in range(256)]
    W = rng.normal(size=(200, 32))
    load = lambda: np.stack([np.array(json.loads(r)) for r in records])
    xb = load()
    t_load = median_time(load)
    t_comp = median_time(lambda: np.tanh(xb @ W).mean())
    print(f"[profiler] batch 256: 데이터 준비 {t_load * 1e3:.2f} ms, 계산 {t_comp * 1e3:.3f} ms "
          f"(데이터 비율 {t_load / (t_load + t_comp):.0%})")

    W1, W2 = rng.normal(size=(4, 8)), rng.normal(size=(8, 2))
    data = rng.normal(size=(512, 4))
    parts = []
    for bs in [1, 16, 256]:
        def epoch():
            for i in range(0, 512, bs):
                np.maximum(data[i:i + bs] @ W1, 0) @ W2
        parts.append(f"batch {bs}: {median_time(epoch, repeat=5) / 512 * 1e6:.2f} µs")
    print("[연산 효율] 샘플당 시간 —", ", ".join(parts))

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 256))
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)
    xt = torch.randn(512, 256)

    def step(bf16):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=bf16):
            loss = net(xt).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    t32 = median_time(lambda: step(False), repeat=10)
    t16 = median_time(lambda: step(True), repeat=10)
    print(f"[mixed precision] CPU 학습 step: float32 {t32 * 1e3:.1f} ms, bf16 autocast {t16 * 1e3:.1f} ms "
          f"(bf16/float32 = {t16 / t32:.2f})")


if __name__ == "__main__":
    main()
