import numpy as np


def smooth(values, weight):
    """보정한 지수이동평균. NaN·inf는 원본 그대로 두고 상태를 바꾸지 않는다.

    values: (T,) 기록된 값, weight: 0 <= w < 1
    반환: (T,) float 배열
    """
    raise NotImplementedError("상태 s와 유한한 값의 개수 k를 갱신하며 s / (1 - w**k)를 계산하세요")
