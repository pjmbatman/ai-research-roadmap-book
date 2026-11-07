import numpy as np


def affine_flow_logpdf(Y, A, b):
    """y = A z + b, z ~ N(0, I_d)일 때 log p_Y(y).

    Y: (N, d), A: (d, d) 가역, b: (d,)
    반환: (N,)
    """
    Y, A, b = (np.asarray(v, dtype=float) for v in (Y, A, b))
    d = A.shape[0]
    Z = np.linalg.solve(A, (Y - b).T).T                          # (N, d) 역변환
    log_pz = -0.5 * np.sum(Z**2, axis=1) - 0.5 * d * np.log(2 * np.pi)   # (N,)
    _, logabsdet = np.linalg.slogdet(A)                          # log|det A|
    return log_pz - logabsdet
