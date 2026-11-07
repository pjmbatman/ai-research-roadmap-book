import numpy as np


def interpolation_barrier(loss, a, b, num=21):
    """두 파라미터 a, b를 잇는 선분 위의 손실과 장벽.

    loss: 파라미터 벡터를 받아 스칼라 손실을 반환하는 함수
    a, b: (P,) 두 해
    반환: (ts, losses, barrier)
      ts      (num,) 0부터 1까지 균등 간격
      losses  (num,) loss((1 - t) a + t b)
      barrier 경로 최대 손실 - max(losses[0], losses[-1])
    """
    raise NotImplementedError("선분 위의 손실과 장벽을 계산하세요")
