import numpy as np


def feasible_pareto(F, G):
    """제약을 만족하는 후보 중 Pareto 최적인 것의 번호를 첫째 목적 오름차순으로 반환한다.

    F: (N, 2) 최소화할 목적, G: (N, m) 제약값(모두 0 이하이면 실현 가능). 반환: (K,) 정수 배열
    """
    raise NotImplementedError("실현 가능한 후보만 남긴 뒤 지배당하지 않는 후보를 고르세요")
