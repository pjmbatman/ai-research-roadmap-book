import numpy as np


def check_column(values, spec):
    """데이터 사전 항목 spec으로 한 열을 검사한다.

    values: (N,) 실수 배열. NaN을 포함할 수 있다
    spec: {"min": float, "max": float, "missing_codes": list[float] (선택)}
    반환: {"missing": int, "out_of_range": int, "valid": int}
    """
    v = np.asarray(values, dtype=float)                       # (N,)
    codes = spec.get("missing_codes", [])
    missing = np.isnan(v) | np.isin(v, codes)                  # (N,)
    out = ~missing & ((v < spec["min"]) | (v > spec["max"]))   # (N,)
    return {
        "missing": int(missing.sum()),
        "out_of_range": int(out.sum()),
        "valid": int(v.size - missing.sum() - out.sum()),
    }
