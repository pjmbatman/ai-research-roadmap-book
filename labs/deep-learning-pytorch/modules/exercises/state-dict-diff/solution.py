import numpy as np


def strip_prefix(state_dict, prefix):
    """prefix로 시작하는 키에서만 접두사를 뗀 새 dict를 반환한다."""
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def diff_state_dict(expected_shapes, state_dict):
    """기대하는 {이름: shape}과 state_dict를 비교한다.

    반환: {"missing": [...], "unexpected": [...], "shape_mismatch": [...]} (각 목록은 정렬)
    """
    want, have = set(expected_shapes), set(state_dict)
    mismatch = [k for k in want & have if tuple(np.shape(state_dict[k])) != tuple(expected_shapes[k])]
    return {
        "missing": sorted(want - have),
        "unexpected": sorted(have - want),
        "shape_mismatch": sorted(mismatch),
    }
