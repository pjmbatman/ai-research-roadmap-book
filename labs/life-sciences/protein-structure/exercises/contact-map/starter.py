import numpy as np


def contact_map(X, cutoff=8.0, min_sep=6):
    """Cα 좌표 X (L, 3)에서 거리 < cutoff 이고 |i-j| >= min_sep 인 쌍의 (L, L) bool 행렬."""
    raise NotImplementedError("거리 행렬을 만들고 두 조건으로 자르세요")


def top_l_precision(scores, true_contacts, min_sep=6):
    """|i-j| >= min_sep 인 쌍 (i < j) 중 점수 상위 L개의 실제 접촉 비율."""
    raise NotImplementedError("상위 L개 쌍을 고르고 정밀도를 계산하세요")
