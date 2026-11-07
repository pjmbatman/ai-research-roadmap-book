import numpy as np


def scores(fc, obs, clim, lat_deg):
    """리드 타임별 위도 가중 RMSE와 ACC.

    fc, obs: (N, L, nlat, nlon), clim: (nlat, nlon), lat_deg: (nlat,)
    반환: rmse (L,), acc (L,)
    """
    raise NotImplementedError("위도 가중 RMSE와 ACC를 계산하세요")
