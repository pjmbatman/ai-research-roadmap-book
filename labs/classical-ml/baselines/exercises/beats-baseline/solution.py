import numpy as np


def paired_verdict(err_model, err_base, n_boot=2000, seed=0):
    """같은 시험 단위 n개에서 잰 두 모델의 오차 (n,), (n,)를 짝지어 비교한다.

    반환: {"skill": float, "diff": float, "ci": (lo, hi), "verdict": "이긴다" | "진다" | "판정 불가"}
    """
    em = np.asarray(err_model, dtype=float)              # (n,)
    eb = np.asarray(err_base, dtype=float)               # (n,)
    d = em - eb                                          # 단위별 짝지은 차이, (n,)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))  # 단위를 복원추출, (B, n)
    boot = d[idx].mean(axis=1)                           # (B,)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    if hi < 0:
        verdict = "이긴다"
    elif lo > 0:
        verdict = "진다"
    else:
        verdict = "판정 불가"
    return {"skill": 1 - em.mean() / eb.mean(), "diff": d.mean(), "ci": (lo, hi), "verdict": verdict}
