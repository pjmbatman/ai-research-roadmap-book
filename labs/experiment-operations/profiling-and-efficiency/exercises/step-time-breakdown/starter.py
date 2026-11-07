import numpy as np


def summarize_steps(steps, warmup):
    """step별 시간 기록에서 warm-up을 빼고 처리량·데이터 대기 비율·겹침 시의 기대 step 시간을 계산한다.

    steps: [{"step", "batch_size", "data_s", "compute_s"}, ...]
    반환: {"throughput", "data_fraction", "overlap_step_s", "overlap_speedup"}
    """
    raise NotImplementedError("앞의 warmup개 step을 빼고 네 값을 계산하세요")
