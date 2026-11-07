import numpy as np


def next_point(x_tr, x_cand, ell, sf=1.0, sn=1e-3):
    """GP 사후분산이 가장 큰 후보를 다음 실행점으로 고른다.

    x_tr: (N,) 이미 실행한 입력, x_cand: (M,) 후보 입력
    ell: 길이척도, sf: 신호 표준편차, sn: 잡음 표준편차
    반환: (x_next, var) — x_next는 스칼라, var는 (M,) 사후분산
    """
    raise NotImplementedError("사후분산 var를 계산하고 가장 큰 후보를 고르세요")
