import numpy as np


def check_column(values, spec):
    """데이터 사전 항목 spec으로 한 열을 검사한다.

    values: (N,) 실수 배열. NaN을 포함할 수 있다
    spec: {"min": float, "max": float, "missing_codes": list[float] (선택)}
    반환: {"missing": int, "out_of_range": int, "valid": int}
    """
    raise NotImplementedError("결측 마스크와 범위 마스크를 만들어 개수를 세세요")
