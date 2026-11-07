import numpy as np


def masked_mae(batches):
    """(pred, target, mask) batch 목록에서 관측된 값 전체의 MAE를 계산한다.

    pred, target, mask: (B, T). mask는 1=관측, 0=padding. None이면 모두 관측.
    관측값이 하나도 없으면 ValueError.
    """
    total, count = 0.0, 0.0
    for pred, target, mask in batches:
        pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
        mask = np.ones_like(pred) if mask is None else np.asarray(mask, dtype=float)
        err = np.where(mask > 0, np.abs(pred - target), 0.0)   # padding의 NaN도 0으로
        total += float((err * mask).sum())
        count += float(mask.sum())
    if count == 0:
        raise ValueError("관측된 값이 없습니다")
    return total / count
