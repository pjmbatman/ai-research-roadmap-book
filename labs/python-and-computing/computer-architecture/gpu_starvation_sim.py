"""5.5 실습: 'GPU 사용률이 낮다'는 현상을 가설별 실험으로 나누어 검증한다(CPU에서 도는 모형).

실행: uv run --project labs python labs/python-and-computing/computer-architecture/gpu_starvation_sim.py

GPU가 없어도 원리를 실험할 수 있도록, 학습 한 step을 네 단계로 흉내 낸다.
  load      : 샘플 읽기·디코딩 (time.sleep으로 디스크 대기 + 작은 NumPy 계산)
  transfer  : 호스트 → 장치 복사 (큰 배열 복사)
  compute   : 장치 계산 (행렬 곱). '장치 사용률' = compute 시간 / 전체 시간
  sync      : 매 step 손실 값을 꺼내 기록 (장치가 끝날 때까지 기다리는 동기화의 모형)
가설마다 한 가지만 바꾼 실험을 돌려, 사용률이 어떻게 변하는지 본다. 시간 값은 기계마다 다르다.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # 장치 계산을 단일 스레드로 고정해 비교를 단순하게 한다
os.environ.setdefault("OMP_NUM_THREADS", "1")

import queue
import threading
import time

import numpy as np

BATCH, D = 32, 512
STEPS = 30
IO_WAIT_S = 0.002          # 샘플 하나를 읽는 대기 시간
SYNC_WAIT_S = 0.004        # 동기화로 장치가 노는 시간(모형)


def load_sample(rng):
    time.sleep(IO_WAIT_S)                      # 디스크·네트워크 대기 (GIL을 놓는다)
    return rng.normal(size=D).astype(np.float32)


def load_batch(rng):
    return np.stack([load_sample(rng) for _ in range(BATCH)])   # (B, D)


def transfer(x, pinned_copies=1):
    for _ in range(pinned_copies):
        x = x.copy()                           # 호스트 → 장치 복사의 모형
    return x


W = np.random.default_rng(0).normal(size=(D, D)).astype(np.float32)


def compute(x, repeats=40):
    for _ in range(repeats):
        x = np.tanh(x @ W)                     # (B, D)
    return x


def run(name, loader_threads=0, cached=False, sync_every=1, compute_repeats=40):
    rng = np.random.default_rng(1)
    cache = load_batch(rng) if cached else None
    q: queue.Queue = queue.Queue(maxsize=8)

    def producer(seed):
        r = np.random.default_rng(seed)
        while True:
            q.put(load_batch(r))

    if loader_threads:
        for k in range(loader_threads):
            threading.Thread(target=producer, args=(100 + k,), daemon=True).start()

    t_compute = 0.0
    t0 = time.perf_counter()
    for step in range(STEPS):
        if cached:
            x = cache
        elif loader_threads:
            x = q.get()                        # 미리 읽어 둔 batch
        else:
            x = load_batch(rng)                # 계산이 데이터 읽기를 기다린다
        x = transfer(x)
        tc = time.perf_counter()
        out = compute(x, compute_repeats)
        t_compute += time.perf_counter() - tc
        if step % sync_every == 0:
            time.sleep(SYNC_WAIT_S)            # loss.item() 같은 동기화의 모형
            float(out.mean())
    wall = time.perf_counter() - t0
    print(f"{name:<34} 전체 {wall:6.3f} s   장치 사용률 {t_compute / wall:5.1%}")


def main():
    print(f"batch {BATCH}, step {STEPS}, 샘플당 대기 {IO_WAIT_S * 1e3:.0f} ms, 동기화 대기 {SYNC_WAIT_S * 1e3:.0f} ms")
    run("기준: 순차 로딩, 매 step 동기화")
    run("가설 1 확인: 데이터를 메모리에 캐시", cached=True)
    run("가설 1 처방: loader 스레드 8개 + prefetch", loader_threads=8)
    run("가설 2 확인: 동기화를 10 step마다", loader_threads=8, sync_every=10)
    run("가설 3 확인: step당 계산량 4배", loader_threads=8, sync_every=10, compute_repeats=160)


if __name__ == "__main__":
    main()
