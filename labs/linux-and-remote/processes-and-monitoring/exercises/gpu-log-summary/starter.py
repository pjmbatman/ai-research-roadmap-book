def summarize(text, idle_util_pct=5, held_mem_mib=1024):
    """nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total
    --format=csv,noheader,nounits 기록을 GPU별로 요약한다.

    반환: {index: {"mean_util": float | None, "max_mem_mib": int, "idle_holding": bool}}
    """
    raise NotImplementedError("GPU별 요약을 계산하세요")
