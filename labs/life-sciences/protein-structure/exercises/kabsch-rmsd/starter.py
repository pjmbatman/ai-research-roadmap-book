import numpy as np


def kabsch_rmsd(P, Q):
    """회전(반사 제외)과 평행이동으로 P를 Q에 최적으로 겹친 뒤의 RMSD.

    P, Q: (L, 3) 좌표 (Å), 같은 행끼리 대응
    반환: float
    """
    raise NotImplementedError("중심 맞추기 → SVD → 회전 → RMSD를 구현하세요")
