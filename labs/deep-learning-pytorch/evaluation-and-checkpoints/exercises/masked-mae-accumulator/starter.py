import numpy as np


def masked_mae(batches):
    """(pred, target, mask) batch 목록에서 관측된 값 전체의 MAE를 계산한다.

    pred, target, mask: (B, T). mask는 1=관측, 0=padding. None이면 모두 관측.
    관측값이 하나도 없으면 ValueError.
    """
    raise NotImplementedError("오차의 합과 관측 수를 따로 누적하세요")
