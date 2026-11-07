import numpy as np


def summarize_steps(steps, warmup):
    """step별 시간 기록에서 warm-up을 빼고 처리량·데이터 대기 비율·겹침 시의 기대 step 시간을 계산한다.

    steps: [{"step", "batch_size", "data_s", "compute_s"}, ...]
    반환: {"throughput", "data_fraction", "overlap_step_s", "overlap_speedup"}
    """
    if warmup >= len(steps):
        raise ValueError("warm-up을 빼면 남는 step이 없습니다")
    kept = steps[warmup:]
    data = np.array([s["data_s"] for s in kept])              # (S,)
    comp = np.array([s["compute_s"] for s in kept])           # (S,)
    n = sum(s["batch_size"] for s in kept)
    total = data + comp                                       # (S,) 지금의 step 시간
    overlap = np.maximum(data, comp)                          # (S,) 완전히 겹칠 때의 step 시간
    return {
        "throughput": float(n / total.sum()),
        "data_fraction": float(data.sum() / total.sum()),
        "overlap_step_s": float(overlap.mean()),
        "overlap_speedup": float(total.mean() / overlap.mean()),
    }
