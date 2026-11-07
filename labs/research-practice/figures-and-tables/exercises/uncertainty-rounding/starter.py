import math


def format_pm(mean, err, sig=2):
    """불확실성 err를 유효숫자 sig자리로, mean을 같은 자릿수로 반올림해 "mean ± err" 문자열을 만든다.

    mean: float, err: float (> 0), sig: int (>= 1)
    반환: str, 예) format_pm(0.41325281, 0.04468845) == "0.413 ± 0.045"
    """
    raise NotImplementedError("자릿수 계산 → err 반올림 → 자릿수 다시 확인 → mean 반올림 → 문자열")
