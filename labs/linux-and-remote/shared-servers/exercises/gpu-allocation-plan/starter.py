def pick_gpus(gpu_csv, apps_csv, n, min_free_mib):
    """비어 있는 GPU 중 번호가 작은 n개를 골라 CUDA_VISIBLE_DEVICES 문자열로 반환한다.

    gpu_csv: "index, uuid, memory.used, memory.total" 줄들 (MiB)
    apps_csv: "gpu_uuid, pid, used_memory" 줄들 (빈 문자열이면 프로세스 없음)
    비어 있는 GPU가 n개보다 적으면 RuntimeError.
    """
    raise NotImplementedError("비어 있는 GPU를 고르세요")
