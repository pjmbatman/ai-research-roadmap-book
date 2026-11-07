import numpy as np


def weighted_mean(field, lat_deg):
    """마지막 두 축 (nlat, nlon)에 대한 cos(위도) 가중 평균. 결측(NaN)은 분자·분모에서 뺀다.

    field: (..., nlat, nlon), lat_deg: (nlat,)
    반환: field.shape[:-2] shape의 배열(2차원 입력이면 스칼라)
    """
    raise NotImplementedError("위도 가중 평균을 계산하세요")
