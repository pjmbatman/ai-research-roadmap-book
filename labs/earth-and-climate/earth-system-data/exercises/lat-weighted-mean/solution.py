import numpy as np


def weighted_mean(field, lat_deg):
    """마지막 두 축 (nlat, nlon)에 대한 cos(위도) 가중 평균. 결측(NaN)은 분자·분모에서 뺀다.

    field: (..., nlat, nlon), lat_deg: (nlat,)
    반환: field.shape[:-2] shape의 배열(2차원 입력이면 스칼라)
    """
    field = np.asarray(field, dtype=float)
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=float)))[:, None]   # (nlat, 1)
    valid = ~np.isnan(field)                                            # (..., nlat, nlon)
    wv = np.where(valid, w, 0.0)                                        # 결측 칸 가중치 0
    num = (wv * np.where(valid, field, 0.0)).sum(axis=(-2, -1))
    den = wv.sum(axis=(-2, -1))
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, num / safe, np.nan)
