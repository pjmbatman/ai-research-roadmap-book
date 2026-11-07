def roofline(flops, bytes_moved, peak_flops_per_s, bandwidth_bytes_per_s):
    """roofline 모형으로 실행 시간 하한, 연산 강도, 병목 종류를 구한다.

    반환: (time_s, intensity, bound) — bound는 "memory" 또는 "compute"
    잘못된 입력에는 ValueError.
    """
    if flops < 0 or bytes_moved < 0 or (flops == 0 and bytes_moved == 0):
        raise ValueError("flops와 bytes_moved는 0 이상이고 둘 다 0일 수 없습니다")
    if peak_flops_per_s <= 0 or bandwidth_bytes_per_s <= 0:
        raise ValueError("장치 성능 값은 양수여야 합니다")
    t_compute = flops / peak_flops_per_s
    t_memory = bytes_moved / bandwidth_bytes_per_s
    intensity = flops / bytes_moved if bytes_moved > 0 else float("inf")
    bound = "memory" if t_memory > t_compute else "compute"
    return max(t_compute, t_memory), intensity, bound
