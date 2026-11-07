import numpy as np


def strip_prefix(state_dict, prefix):
    """prefix로 시작하는 키에서만 접두사를 뗀 새 dict를 반환한다."""
    raise NotImplementedError("키마다 prefix로 시작하는지 확인하세요")


def diff_state_dict(expected_shapes, state_dict):
    """기대하는 {이름: shape}과 state_dict를 비교한다.

    반환: {"missing": [...], "unexpected": [...], "shape_mismatch": [...]} (각 목록은 정렬)
    """
    raise NotImplementedError("두 키 집합을 비교하고 shape을 확인하세요")
