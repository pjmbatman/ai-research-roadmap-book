import numpy as np


def rk4_step(f, t, y, h):
    """고전적 RK4로 y' = f(t, y)를 한 step 전진한다.

    f: f(t, y) -> (d,) 배열
    t: 현재 시각 (float)
    y: 현재 상태, shape (d,) (리스트도 허용)
    h: step 크기
    반환: 다음 상태 y_{n+1}, shape (d,)
    """
    raise NotImplementedError("k1, k2, k3, k4를 계산해 가중 평균으로 전진하세요")
