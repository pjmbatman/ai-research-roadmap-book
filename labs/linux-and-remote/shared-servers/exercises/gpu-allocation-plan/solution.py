def pick_gpus(gpu_csv, apps_csv, n, min_free_mib):
    """비어 있는 GPU 중 번호가 작은 n개를 골라 CUDA_VISIBLE_DEVICES 문자열로 반환한다.

    gpu_csv: "index, uuid, memory.used, memory.total" 줄들 (MiB)
    apps_csv: "gpu_uuid, pid, used_memory" 줄들 (빈 문자열이면 프로세스 없음)
    비어 있는 GPU가 n개보다 적으면 RuntimeError.
    """
    busy_uuids = set()
    for line in apps_csv.splitlines():
        if line.strip():
            busy_uuids.add(line.split(",")[0].strip())

    free = []
    for line in gpu_csv.splitlines():
        if not line.strip():
            continue
        index, uuid, used, total = [field.strip() for field in line.split(",")]
        if uuid not in busy_uuids and int(total) - int(used) >= min_free_mib:
            free.append(int(index))

    free.sort()
    if len(free) < n:
        raise RuntimeError(f"비어 있는 GPU {len(free)}개, 요청 {n}개")
    return ",".join(str(i) for i in free[:n])
