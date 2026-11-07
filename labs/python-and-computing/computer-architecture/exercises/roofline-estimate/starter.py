def roofline(flops, bytes_moved, peak_flops_per_s, bandwidth_bytes_per_s):
    """roofline 모형으로 실행 시간 하한, 연산 강도, 병목 종류를 구한다.

    반환: (time_s, intensity, bound) — bound는 "memory" 또는 "compute"
    잘못된 입력에는 ValueError.
    """
    raise NotImplementedError("연산 시간과 메모리 시간을 비교하세요")
