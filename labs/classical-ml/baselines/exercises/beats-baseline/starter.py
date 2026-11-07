import numpy as np


def paired_verdict(err_model, err_base, n_boot=2000, seed=0):
    """같은 시험 단위 n개에서 잰 두 모델의 오차 (n,), (n,)를 짝지어 비교한다.

    반환: {"skill": float, "diff": float, "ci": (lo, hi), "verdict": "이긴다" | "진다" | "판정 불가"}
    """
    raise NotImplementedError("짝지은 차이를 부트스트랩해 판정하세요")
