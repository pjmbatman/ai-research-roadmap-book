import numpy as np


def check_standardize(standardize):
    """standardize 구현이 명세를 어기면 AssertionError를 낸다.

    명세: x (N, D) → 새 배열 (N, D). 각 열의 평균 0, 표준편차(ddof=0) 1.
    값이 모두 같은 열은 0. 입력 x는 바꾸지 않는다.
    """
    raise NotImplementedError("테스트 데이터를 만들고, 명세의 각 문장을 assert로 확인하세요")
