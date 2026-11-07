"""12.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/statistical-inference/intervals-and-testing/cells.py
"""
# %% cell 1 — 신뢰구간의 포함 확률
import numpy as np
from math import erf, sqrt

rng = np.random.default_rng(0)
t975 = {9: 2.262, 29: 2.045}           # t 분포의 97.5% 분위수 (자유도 n-1)

def coverage(sample, mean_true, n, R=20000):
    X = sample((R, n))                                    # 같은 실험을 R번 반복, (R, n)
    m, se = X.mean(axis=1), X.std(axis=1, ddof=1) / np.sqrt(n)
    half = t975[n - 1] * se                               # 구간 반폭, (R,)
    return np.mean((m - half <= mean_true) & (mean_true <= m + half))

for n in (10, 30):
    c_norm = coverage(lambda s: rng.normal(5.0, 2.0, s), 5.0, n)
    c_skew = coverage(lambda s: rng.lognormal(0.0, 1.0, s), np.exp(0.5), n)
    print(f"n = {n}:  정규 데이터 {c_norm:.3f}   로그정규 데이터 {c_skew:.3f}   (목표 0.950)")

# %% cell 2 — p-value의 분포, 1종 오류, 검정력
Phi = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))   # 표준정규 CDF

def z_test_p(a, b):                      # σ = 1을 아는 두 표본 z 검정, 양측 p-value
    z = (b.mean(axis=1) - a.mean(axis=1)) / np.sqrt(2 / a.shape[1])
    return 2 * (1 - Phi(np.abs(z)))

R = 20000
for d, n in ((0.0, 20), (0.5, 20), (0.5, 64)):           # d: 참 효과 크기 (σ 단위)
    p = z_test_p(rng.normal(0, 1, (R, n)), rng.normal(d, 1, (R, n)))
    theory = Phi(d * np.sqrt(n / 2) - 1.96) + Phi(-d * np.sqrt(n / 2) - 1.96)
    print(f"d = {d}, n = {n:2d}:  P(p < 0.05) = {np.mean(p < 0.05):.3f}  (이론 {theory:.3f})"
          f"   P(p < 0.01) = {np.mean(p < 0.01):.3f}")

# %% cell 3 — 통계적 유의성과 효과 크기
for d, n in ((0.02, 100000), (0.5, 20)):
    a, b = rng.normal(0, 1, n), rng.normal(d, 1, n)        # 그룹당 n개
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    p = 2 * (1 - Phi(abs(diff / se)))
    print(f"n = {n:6d}:  차이 {diff:+.3f}  95% CI [{diff - 1.96 * se:+.3f}, {diff + 1.96 * se:+.3f}]"
          f"  p = {p:.1e}  Cohen's d = {diff / pooled_sd:.2f}")

# %% cell 4 — 부트스트랩: 중앙값의 표준오차와 percentile 구간
life = rng.lognormal(np.log(800), 0.4, size=40)           # 셀 40개의 사이클 수명, (n,)
B = 5000
idx = rng.integers(0, len(life), size=(B, len(life)))     # 복원추출 인덱스, (B, n)
boot = np.median(life[idx], axis=1)                       # 부트스트랩 중앙값, (B,)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"표본 중앙값 {np.median(life):.0f} 사이클,  부트스트랩 SE {boot.std():.0f},  95% percentile 구간 [{lo:.0f}, {hi:.0f}]")

pop = np.median(rng.lognormal(np.log(800), 0.4, size=(B, 40)), axis=1)   # 모집단에서 새로 뽑은 표본들
print(f"모집단에서 반복 추출한 중앙값의 표준편차 {pop.std():.0f}  (부트스트랩이 흉내 내려는 값)")

# %% cell 5 — 다중 비교: Bonferroni와 Benjamini–Hochberg
m, m1, R = 100, 10, 2000                       # 검정 100개, 그중 실제 효과 10개, 반복 R번
mu = np.r_[np.full(m1, 3.0), np.zeros(m - m1)]
p = 2 * (1 - Phi(np.abs(rng.normal(mu, 1.0, size=(R, m)))))   # (R, m)

def bh(p, q=0.05):                             # 행마다 Benjamini–Hochberg 기각 여부
    order = np.argsort(p, axis=1)
    ok = np.take_along_axis(p, order, axis=1) <= q * np.arange(1, p.shape[1] + 1) / p.shape[1]
    k = np.where(ok.any(axis=1), p.shape[1] - np.argmax(ok[:, ::-1], axis=1), 0)   # 통과한 최대 순위
    rej = np.zeros_like(p, dtype=bool)
    np.put_along_axis(rej, order, np.arange(p.shape[1]) < k[:, None], axis=1)
    return rej

for name, rej in (("보정 없음", p < 0.05), ("Bonferroni", p < 0.05 / m), ("BH (q=0.05)", bh(p))):
    false = rej[:, m1:].sum(axis=1)
    fdp = false / np.maximum(rej.sum(axis=1), 1)
    print(f"{name:11s}  발견 {rej.sum(axis=1).mean():5.2f}  참 발견 {rej[:, :m1].sum(axis=1).mean():4.2f}"
          f"  P(거짓 발견 ≥ 1) {np.mean(false > 0):.3f}  평균 FDP {fdp.mean():.3f}")
