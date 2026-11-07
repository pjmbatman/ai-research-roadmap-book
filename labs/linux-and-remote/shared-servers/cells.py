"""2.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linux-and-remote/shared-servers/cells.py
"""
# %% cell 1 — 비어 있는 GPU 고르기
import io

import pandas as pd

# nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits (가상의 출력)
gpu_csv = "0, 30512, 40960\n1, 38200, 40960\n2, 4, 40960\n3, 4, 40960"
# nvidia-smi --query-compute-apps=pid,used_memory 를 ps로 소유자와 GPU 번호까지 붙였다고 가정한 표
procs = pd.DataFrame({"gpu": [0, 1], "pid": [5310, 4190], "user": ["lee", "kim"], "mem_mib": [30500, 38190]})

gpu = pd.read_csv(io.StringIO(gpu_csv), names=["index", "used_mib", "total_mib"], skipinitialspace=True)
gpu["free_mib"] = gpu["total_mib"] - gpu["used_mib"]
gpu["n_procs"] = gpu["index"].map(procs["gpu"].value_counts()).fillna(0).astype(int)
free = gpu[(gpu["n_procs"] == 0) & (gpu["free_mib"] >= 20000)]
print(gpu.to_string(index=False))
print("CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in free["index"]))

# %% cell 2 — array job: task id → seed, 결과 모으기
import numpy as np

seeds = [0, 1, 2, 3, 4]                       # --array=0-4 의 각 SLURM_ARRAY_TASK_ID가 고르는 seed
def run_task(task_id):                        # 실제로는 train.py가 하는 일(가상의 정확도)
    rng = np.random.default_rng(seeds[task_id])
    return 0.90 + 0.02 * rng.standard_normal()

results = {f"seed{seeds[t]}": run_task(t) for t in range(5)}
acc = np.array(list(results.values()))        # (5,)
print({k: round(v, 4) for k, v in results.items()})
print(f"정확도 평균 {acc.mean():.4f} ± 표준편차 {acc.std(ddof=1):.4f} (n = {len(acc)})")

# %% cell 3 — checkpoint 주기와 선점: 총 소요 시간의 Monte Carlo
def wall_time_h(interval_h, work_h=10.0, ckpt_h=0.05, mtbf_h=8.0, restart_h=0.1, rng=None):
    done, t = 0.0, 0.0
    while done < work_h:
        seg = min(interval_h, work_h - done)              # 다음 checkpoint까지 할 일
        fail_in = rng.exponential(mtbf_h)                 # 다음 선점까지 남은 시간
        if fail_in >= seg + ckpt_h:                       # checkpoint까지 무사히 마침
            t, done = t + seg + ckpt_h, done + seg
        else:                                             # 선점: 마지막 checkpoint 이후 작업을 잃음
            t += fail_in + restart_h
    return t

rng = np.random.default_rng(0)
for interval_h in [0.25, 0.5, 0.9, 2.0, 4.0, 10.0]:
    runs = [wall_time_h(interval_h, rng=rng) for _ in range(2000)]
    print(f"주기 {interval_h:5.2f} h  평균 총 소요 {np.mean(runs):6.2f} h")
