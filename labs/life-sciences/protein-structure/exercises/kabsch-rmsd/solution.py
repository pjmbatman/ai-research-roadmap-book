import numpy as np


def kabsch_rmsd(P, Q):
    """회전(반사 제외)과 평행이동으로 P를 Q에 최적으로 겹친 뒤의 RMSD.

    P, Q: (L, 3) 좌표 (Å), 같은 행끼리 대응
    반환: float
    """
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    P0, Q0 = P - P.mean(0), Q - Q.mean(0)              # (L, 3)
    U, _, Vt = np.linalg.svd(P0.T @ Q0)                # (3, 3)
    d = np.sign(np.linalg.det(U @ Vt))                 # 반사 방지
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return float(np.sqrt(np.mean(np.sum((P0 @ R - Q0) ** 2, axis=1))))
