import numpy as np


def to_si(record: dict, passthrough=()) -> dict:
    """이름의 단위 접미사를 읽어 값을 SI 계열 단위(v, a, s, ah, k)로 바꾼 새 dict를 반환한다.

    record: 이름 → 숫자 또는 NumPy 배열. 바꾸지 않는다.
    passthrough: 단위가 없어 그대로 둘 이름들
    알 수 없는 접미사나 변환 후 이름 충돌은 ValueError.
    """
    raise NotImplementedError("단위를 변환하세요")
