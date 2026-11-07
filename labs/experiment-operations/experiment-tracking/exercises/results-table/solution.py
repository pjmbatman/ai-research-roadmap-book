import math

import numpy as np


def summarize(runs, group_by, metric):
    """run 기록 리스트를 group_by 설정 키로 묶어 metric을 요약한다.

    runs: [{"run_id", "config", "status", "metrics"}, ...]
    group_by: 설정 키 리스트, 예 ["lr"]
    반환: {(값, ...): {"mean", "std", "n", "missing", "run_ids"}}
    """
    groups = {}
    for r in runs:
        key = tuple(r["config"][k] for k in group_by)
        groups.setdefault(key, []).append(r)
    table = {}
    for key, members in groups.items():
        done = [r for r in members if r["status"] == "finished"]
        values = np.array([r["metrics"][metric] for r in done], dtype=float)   # (n,)
        table[key] = {
            "mean": float(values.mean()) if len(values) else math.nan,
            "std": float(values.std(ddof=1)) if len(values) >= 2 else math.nan,
            "n": len(done),
            "missing": len(members) - len(done),
            "run_ids": sorted(r["run_id"] for r in done),
        }
    return table
