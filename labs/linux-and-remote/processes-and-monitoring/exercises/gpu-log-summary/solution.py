def summarize(text, idle_util_pct=5, held_mem_mib=1024):
    """nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total
    --format=csv,noheader,nounits 기록을 GPU별로 요약한다.

    반환: {index: {"mean_util": float | None, "max_mem_mib": int, "idle_holding": bool}}
    """
    utils, mems = {}, {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("timestamp"):
            continue
        _, idx, util, mem_used, _ = [field.strip() for field in line.split(",")]
        idx = int(idx)
        utils.setdefault(idx, [])
        mems.setdefault(idx, [])
        if util != "[N/A]":
            utils[idx].append(float(util))
        if mem_used != "[N/A]":
            mems[idx].append(int(mem_used))

    result = {}
    for idx in sorted(utils):
        mean_util = sum(utils[idx]) / len(utils[idx]) if utils[idx] else None
        max_mem = max(mems[idx]) if mems[idx] else 0
        idle = mean_util is not None and mean_util < idle_util_pct and max_mem > held_mem_mib
        result[idx] = {"mean_util": mean_util, "max_mem_mib": max_mem, "idle_holding": idle}
    return result
