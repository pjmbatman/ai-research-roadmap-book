"""2.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linux-and-remote/processes-and-monitoring/cells.py
"""
# %% cell 1 — ps 출력 파싱: 사용자별 메모리, zombie, 부모가 1인 프로세스
import io

import pandas as pd

# ps -eo pid,ppid,user,stat,rss,etime,comm --sort=-rss 형식을 흉내 낸 가상의 출력 (RSS 단위 KiB)
ps_txt = """  PID  PPID USER STAT     RSS     ELAPSED COMMAND
 4121     1 kim  Sl   9437184  2-03:11:05 python
 5310  5001 lee  Rl   7340032    05:12:40 python
 4188  4121 kim  S    3145728  2-03:10:58 python
 4190     1 kim  S    3145728  6-21:40:02 python
 6001  5990 park Z          0       00:10 python"""
ps = pd.read_csv(io.StringIO(ps_txt), sep=r"\s+")
ps["rss_gib"] = ps["RSS"] / 1024**2                     # KiB → GiB
print(ps.groupby("USER")["rss_gib"].sum().round(1).to_string())
print("zombie(Z):", ps.loc[ps["STAT"].str.startswith("Z"), "PID"].tolist())
print("PPID = 1:", ps.loc[ps["PPID"] == 1, ["PID", "ELAPSED"]].values.tolist())

# %% cell 2 — nvidia-smi --query-gpu 출력 파싱
# nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw
#            --format=csv,noheader,nounits 형식을 흉내 낸 가상의 출력
gpu_csv = """0, 98, 30512, 40960, 71, 245.3
1, 0, 38200, 40960, 38, 61.2
2, 35, 12004, 40960, 55, 120.8
3, 0, 4, 40960, 30, 52.0"""
cols = ["index", "util_pct", "mem_used_mib", "mem_total_mib", "temp_c", "power_w"]
gpu = pd.read_csv(io.StringIO(gpu_csv), names=cols, skipinitialspace=True)
gpu["mem_free_mib"] = gpu["mem_total_mib"] - gpu["mem_used_mib"]
gpu["idle_holding"] = (gpu["util_pct"] < 5) & (gpu["mem_used_mib"] > 1024)   # 메모리만 잡고 노는 GPU
print(gpu[["index", "util_pct", "mem_free_mib", "temp_c", "idle_holding"]].to_string(index=False))

# %% cell 3 — '사용률'은 무엇을 재는가: 단순화한 시간선 모델
import numpy as np

def simulate(load_ms, kernel_ms, sm_frac, steps=200):
    """한 step = 데이터 대기(load_ms, GPU idle) + 커널 실행(kernel_ms, SM의 sm_frac만 사용). 1 ms 간격."""
    step = np.r_[np.zeros(load_ms), np.full(kernel_ms, sm_frac)]     # (load_ms + kernel_ms,)
    timeline = np.tile(step, steps)                                   # (T,)
    util = np.mean(timeline > 0)          # 커널이 실행 중이던 시간의 비율(nvidia-smi 사용률의 정의)
    work = np.mean(timeline)              # 실제로 SM을 쓴 비율(이 모델에서만 알 수 있는 값)
    return util, work

cases = {"데이터 대기형": (60, 40, 0.9), "작은 커널형": (0, 100, 0.1), "건강한 학습": (2, 98, 0.9)}
for name, args in cases.items():
    util, work = simulate(*args)
    print(f"{name:7s}  사용률 {util:5.0%}   실제 SM 활용 {work:5.0%}")
