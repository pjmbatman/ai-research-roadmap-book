"""23.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-practice/experimental-design/cells.py
"""
# %% cell 1 — 검정력 분석: 몇 개의 seed가 필요한가
import numpy as np
from scipy import stats

delta = 0.01                     # 검출하고 싶은 최소 개선 (정확도)
sd_run, rho = 0.02, 0.8          # 실행 간 표준편차, 같은 분할에서 두 방법 점수의 상관

def power(n, paired):
    if paired:                   # 같은 분할·seed에서 짝지은 차이: sd = sd_run * sqrt(2(1 - rho))
        sd_d, df = sd_run * np.sqrt(2 * (1 - rho)), n - 1
        nc = delta / (sd_d / np.sqrt(n))
    else:                        # 방법마다 따로 n번: 차이의 표준오차 = sd_run * sqrt(2/n)
        df, nc = 2 * n - 2, delta / (sd_run * np.sqrt(2 / n))
    t = stats.t.ppf(0.975, df)
    return stats.nct.sf(t, df, nc) + stats.nct.cdf(-t, df, nc)   # 양측 검정의 검정력

for n in (5, 10, 20, 40, 70):
    print(f"실행 {n:>2}개/방법  짝짓지 않음 {power(n, False):.2f}   짝지음 {power(n, True):.2f}")

rng = np.random.default_rng(0)   # 시뮬레이션으로 확인: n = 10, 짝지음
split = rng.normal(0, sd_run * np.sqrt(rho), size=(20000, 10))            # 분할(블록) 효과
a = split + rng.normal(0, sd_run * np.sqrt(1 - rho), size=(20000, 10))
b = split + delta + rng.normal(0, sd_run * np.sqrt(1 - rho), size=(20000, 10))
print(f"시뮬레이션 (n = 10, 짝지음): {np.mean(stats.ttest_rel(b, a, axis=1).pvalue < 0.05):.2f}")

# %% cell 2 — 탐색 예산이 다르면 같은 방법도 달라 보인다
def tuned_test_score(budget, rng):
    quality = rng.normal(0.80, 0.02, size=budget)            # 설정마다 다른 참 성능, (budget,)
    val = quality + rng.normal(0, 0.01, size=budget)         # validation 점수 (잡음 포함)
    best = np.argmax(val)
    return quality[best] + rng.normal(0, 0.01)               # 고른 설정의 test 점수

rng = np.random.default_rng(1)
for b_base, b_prop in [(5, 5), (5, 50), (50, 50)]:
    gap = [tuned_test_score(b_prop, rng) - tuned_test_score(b_base, rng) for _ in range(4000)]
    print(f"탐색 예산 baseline {b_base:>2}회, 제안 {b_prop:>2}회 → 평균 test 차이 {np.mean(gap):+.4f}  (두 방법은 동일)")

# %% cell 3 — 2^3 요인 설계: 구조·학습 recipe·데이터 증강의 효과 분리
import itertools

levels = np.array(list(itertools.product([0, 1], repeat=3)))   # (8, 3): 구조, recipe, 증강
def run(arch, recipe, aug, rng):                               # 가상의 실험 결과 (정확도)
    return (0.80 + 0.002 * arch + 0.020 * recipe + 0.006 * aug
            + 0.010 * arch * recipe + rng.normal(0, 0.004))

rng = np.random.default_rng(2)
reps = 3                                                       # 조합마다 seed 3개
y = np.array([[run(*lv, rng) for _ in range(reps)] for lv in levels]).mean(1)   # (8,)
S = 2 * levels - 1                                             # -1/+1 부호, (8, 3)
for j, name in enumerate(["구조", "recipe", "증강"]):
    print(f"{name:<6} 주효과 {y @ S[:, j] / 4:+.4f}")
print(f"구조×recipe 교호작용 {y @ (S[:, 0] * S[:, 1]) / 4:+.4f}")
new, old = y[(levels == [1, 1, 0]).all(1)][0], y[(levels == [0, 0, 0]).all(1)][0]
print(f"'새 구조 + 새 recipe' vs '옛 구조 + 옛 recipe': {new - old:+.4f}  ← 전부 구조 덕으로 보고하기 쉽다")
print(f"옛 recipe에서 구조만 바꾼 효과: {y[(levels == [1, 0, 0]).all(1)][0] - old:+.4f}")
