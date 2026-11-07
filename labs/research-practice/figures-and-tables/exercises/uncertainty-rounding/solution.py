import math


def format_pm(mean, err, sig=2):
    """불확실성 err를 유효숫자 sig자리로, mean을 같은 자릿수로 반올림해 "mean ± err" 문자열을 만든다.

    mean: float, err: float (> 0), sig: int (>= 1)
    반환: str, 예) format_pm(0.41325281, 0.04468845) == "0.413 ± 0.045"
    """
    if not err > 0:
        raise ValueError(f"불확실성은 양수여야 합니다: {err}")
    dec = sig - 1 - math.floor(math.log10(err))           # err의 유효숫자 sig개가 남는 소수 자릿수
    err_r = round(err, dec)
    dec = sig - 1 - math.floor(math.log10(err_r))         # 반올림으로 자릿수가 올라갔으면 다시 정한다
    err_r = round(err, dec)
    mean_r = round(mean, dec) + 0.0                       # + 0.0: 음의 0을 0으로
    d = max(dec, 0)
    return f"{mean_r:.{d}f} ± {err_r:.{d}f}"
