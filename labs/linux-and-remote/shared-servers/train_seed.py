"""2.5 실습: array job의 한 task가 실행하는 작은 '학습'. checkpoint에서 재시작하고, SIGTERM·SIGUSR1을 받으면 저장 후 종료한다.

실행(로컬에서 Slurm 흉내): SLURM_ARRAY_TASK_ID=0 python3 train_seed.py --out runs
Slurm에서는 array_job.sh가 실행한다.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

SEEDS = [0, 1, 2, 3, 4]
stop_requested = False


def on_signal(signum, frame):
    global stop_requested
    stop_requested = True                    # 현재 step을 마친 뒤 저장하고 끝낸다


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--step-sleep-s", type=float, default=0.0)   # 선점 실험용으로 일부러 느리게
    args = p.parse_args()

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    seed = SEEDS[task_id]
    run_dir = Path(args.out) / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "ckpt.npz"

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(256, 8))                          # (N, D)
    w_true = np.arange(1.0, 9.0)                           # (D,)
    y = X @ w_true + 0.5 * rng.normal(size=256)            # (N,)

    if ckpt.exists():                                      # 재시작: 마지막 checkpoint에서 이어 간다
        state = np.load(ckpt)
        w, start = state["w"], int(state["step"])
        print(f"[seed {seed}] checkpoint step {start}에서 재시작", flush=True)
    else:
        w, start = np.zeros(8), 0

    for step in range(start, args.steps):
        grad = X.T @ (X @ w - y) / len(y)                  # (D,)
        w = w - 0.01 * grad
        time.sleep(args.step_sleep_s)
        if (step + 1) % args.ckpt_every == 0 or stop_requested:
            tmp = run_dir / "ckpt.tmp.npz"
            np.savez(tmp, w=w, step=step + 1)
            os.replace(tmp, ckpt)                          # 원자적 교체: 저장 도중 죽어도 이전 checkpoint는 온전
        if stop_requested:
            print(f"[seed {seed}] 종료 요청: step {step + 1}에서 저장하고 종료", flush=True)
            sys.exit(0)

    mse = float(np.mean((X @ w - y) ** 2))
    (run_dir / "result.json").write_text(json.dumps({"seed": seed, "steps": args.steps, "mse": mse}))
    print(f"[seed {seed}] 완료 mse={mse:.4f}", flush=True)


if __name__ == "__main__":
    main()
