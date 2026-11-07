import numpy as np


def scores(fc, obs, clim, lat_deg):
    """리드 타임별 위도 가중 RMSE와 ACC.

    fc, obs: (N, L, nlat, nlon), clim: (nlat, nlon), lat_deg: (nlat,)
    반환: rmse (L,), acc (L,)
    """
    fc, obs, clim = (np.asarray(a, dtype=float) for a in (fc, obs, clim))
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=float)))[:, None]   # (nlat, 1)
    w = w / (w * np.ones_like(clim)).mean()                             # 격자 평균이 1
    mse = (w * (fc - obs) ** 2).mean(axis=(-2, -1))                     # (N, L)
    rmse = np.sqrt(mse.mean(axis=0))                                    # (L,)
    fa, oa = fc - clim, obs - clim                                      # 편차
    num = (w * fa * oa).sum(axis=(-2, -1))
    den = np.sqrt((w * fa**2).sum(axis=(-2, -1)) * (w * oa**2).sum(axis=(-2, -1)))
    acc = (num / den).mean(axis=0)                                      # (L,)
    return rmse, acc
