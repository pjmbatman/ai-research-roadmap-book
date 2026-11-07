"""21.6 실습: 이상 탐지 모델의 threshold를 비용으로 정하고, 불확실할 때 판단을 보류하는 규칙을 평가한다.

실행: uv run --project labs python labs/reliability/decision-making/alarm_policy.py

합성 데이터(설비 센서를 흉내 낸 것으로, 실제 설비 데이터가 아니다):
  특징 6개(진동·온도·전류 등, 표준화), 고장률 약 5%. 고장이면 특징 3개의 평균이 이동한다.
  학습 10,000 / 검증 10,000 / 시험 20,000. 모델: scikit-learn LogisticRegression.
비용: 오경보(불필요한 점검) 1, 미탐지(놓친 고장) 25, 보류(사람이 데이터를 검토) 0.6.
정책
  A  threshold 0.5
  B  비용으로 정한 이론 threshold c_FP / (c_FP + c_FN)
  C  검증 데이터에서 비용을 최소화한 threshold
  D  C + 보류 구간 [lo, hi]: 그 안이면 사람이 검토(검토하면 정답을 안다고 가정)
마지막 줄은 센서 하나가 0.5만큼 어긋난(drift) 시험 데이터에서 같은 정책 D의 비용이다.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
C_FP, C_FN, C_REVIEW = 1.0, 25.0, 0.6


def sample(n, drift=0.0):
    y = rng.uniform(size=n) < 0.05
    X = rng.normal(size=(n, 6))
    X[y, :3] += np.array([1.2, 0.9, 0.7])
    X[:, 0] += drift
    return X, y


Xtr, ytr = sample(10000)
Xva, yva = sample(10000)
Xte, yte = sample(20000)
model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
pva, pte = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]


def cost(p, y, thr, band=None):
    alarm = p > thr
    c = C_FP * (alarm & ~y) + C_FN * (~alarm & y)
    if band is not None:
        review = (p >= band[0]) & (p <= band[1])
        c = np.where(review, C_REVIEW, c)
        return c.mean(), review.mean()
    return c.mean(), 0.0


grid = np.round(np.linspace(0.005, 0.6, 120), 3)
thr_c = grid[np.argmin([cost(pva, yva, g)[0] for g in grid])]
bands = [(lo, hi) for lo in grid[:40] for hi in grid if hi > lo]
band_d = min(bands, key=lambda b: cost(pva, yva, thr_c, b)[0])

# 보정 점검: 시험 데이터에서 예측 확률 구간별 관측 빈도
bins = np.array([0, 0.02, 0.05, 0.1, 0.3, 1.0])
idx = np.digitize(pte, bins) - 1
print("보정 점검 (시험): 구간  평균 예측  관측 빈도  개수")
for k in range(len(bins) - 1):
    sel = idx == k
    print(f"  [{bins[k]:.2f}, {bins[k + 1]:.2f})  {pte[sel].mean():.3f}     {yte[sel].mean():.3f}   {sel.sum():5d}")

print("\n정책                         threshold   사례당 비용   보류율   미탐지율(고장 중)")
for name, thr, band in (("A  0.5", 0.5, None), ("B  이론 c_FP/(c_FP+c_FN)", C_FP / (C_FP + C_FN), None),
                        ("C  검증에서 최적", thr_c, None), (f"D  C + 보류 [{band_d[0]:.3f}, {band_d[1]:.3f}]", thr_c, band_d)):
    c, r = cost(pte, yte, thr, band)
    missed = (pte <= thr) & yte
    if band is not None:
        missed &= ~((pte >= band[0]) & (pte <= band[1]))
    print(f"{name:28s} {thr:8.3f}    {c:9.4f}   {r:6.3f}   {missed.sum() / yte.sum():.3f}")

Xdr, ydr = sample(20000, drift=0.5)
pdr = model.predict_proba(Xdr)[:, 1]
c, r = cost(pdr, ydr, thr_c, band_d)
print(f"\n센서 0 drift +0.5 후 정책 D: 사례당 비용 {c:.4f}, 보류율 {r:.3f}, 경보율 {np.mean(pdr > thr_c):.3f} (drift 전 {np.mean(pte > thr_c):.3f})")
