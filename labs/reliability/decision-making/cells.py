"""21.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/reliability/decision-making/cells.py
"""
# %% cell 1 — 예측분포로 기대 효용 최대화: 어떤 후보를 합성할까
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
N = 2000
mu = rng.normal(0.0, 1.0, N)                    # 후보 물질의 예측 평균 (표준화된 물성)
sd = rng.uniform(0.2, 1.5, N)                   # 예측 표준편차: 후보마다 다르다
y = mu + sd * rng.normal(size=N)                # 실제 물성 (예측분포가 보정되어 있다고 가정)
t, gain, cost = 1.5, 10.0, 1.0                  # 목표 물성 t를 넘으면 +10, 합성 비용 1

p_hit = norm.sf(t, mu, sd)                      # P(y > t), (N,)
rules = {
    "평균이 목표 이상 (mu > t)": mu > t,
    "기대 효용 > 0 (P > 0.1)": gain * p_hit - cost > 0,
    "모두 합성": np.ones(N, bool),
}
for name, go in rules.items():
    hits = np.sum(go & (y > t))
    utility = gain * hits - cost * go.sum()
    print(f"{name:22s} 합성 {go.sum():4d}개  성공 {hits:3d}개  총 효용 {utility:6.0f}")

# %% cell 2 — 비대칭 비용의 연속 결정: 예측분포의 분위수를 고른다
m = 20000
heat_mu_kw = rng.uniform(5.0, 15.0, m)          # 장비별 예측 발열량 평균 (kW)
heat_sd_kw = rng.uniform(0.5, 3.0, m)           # 예측 표준편차: 장비마다 다르다
heat_kw = heat_mu_kw + heat_sd_kw * rng.normal(size=m)   # 실제 발열량
C_UNDER, C_OVER = 9.0, 1.0                      # 냉각 부족 1 kW당 9, 과잉 1 kW당 1
tau = C_UNDER / (C_UNDER + C_OVER)              # 최적 분위수 0.9

def mean_cost(cap_kw):
    short = np.maximum(heat_kw - cap_kw, 0)
    return np.mean(C_UNDER * short + C_OVER * np.maximum(cap_kw - heat_kw, 0))

z = norm.ppf(tau)
for name, cap in (("평균으로 설계", heat_mu_kw),
                  ("0.9 분위수로 설계", heat_mu_kw + z * heat_sd_kw),
                  ("0.9 분위수, sd를 절반으로 과소추정", heat_mu_kw + z * 0.5 * heat_sd_kw),
                  ("평균 + 고정 여유 (같은 평균 용량)", heat_mu_kw + z * heat_sd_kw.mean())):
    print(f"{name:24s} 평균 용량 {cap.mean():5.2f} kW   평균 비용 {mean_cost(cap):.3f}")

# %% cell 3 — 오경보·미탐지 비용으로 정하는 threshold
sig = lambda z: 1 / (1 + np.exp(-z))
s = rng.normal(size=40000)                      # 이상 점수의 근거가 되는 신호
logit = -3.5 + 1.5 * s                          # 참 로짓: 고장률 약 7%
fault = rng.uniform(size=s.size) < sig(logit)
C_FP, C_FN = 1.0, 20.0                          # 불필요한 점검 1, 놓친 고장 20
val, te = slice(0, 20000), slice(20000, None)

def cost_per_1000(p, thr, idx):
    alarm = p[idx] > thr
    return 1000 * np.mean(C_FP * (alarm & ~fault[idx]) + C_FN * (~alarm & fault[idx]))

models = {"보정된 모델": sig(logit), "보정이 틀린 모델": sig(2.5 * logit + 3.0)}
thr_bayes = C_FP / (C_FP + C_FN)
grid = np.linspace(0.001, 0.6, 600)
for name, p in models.items():
    thr_val = grid[np.argmin([cost_per_1000(p, g, val) for g in grid])]   # 검증 데이터로 고른 threshold
    row = "  ".join(f"thr {thr:.3f}: {cost_per_1000(p, thr, te):5.1f}" for thr in (0.5, thr_bayes, thr_val))
    print(f"{name:9s} 1000대당 비용  {row}")

# %% cell 4 — 선택적 예측: coverage–risk와 집단별 보류율
n = 20000
group_b = rng.uniform(size=n) < 0.2             # 집단 B: 전체의 20%, 신호가 약하다
signal = rng.normal(size=n) * np.where(group_b, 0.8, 2.5)
p1 = sig(signal)                                # 보정된 class-1 확률
label = rng.uniform(size=n) < p1
conf = np.maximum(p1, 1 - p1)                   # 예측 class의 확률
wrong = (p1 > 0.5) != label

print("coverage  오류율   집단A 응답률  집단B 응답률")
for cov in (1.0, 0.8, 0.6):
    answer = conf >= np.quantile(conf, 1 - cov)  # 확신이 큰 순서로 cov 비율만 답한다
    print(f"  {cov:.1f}     {wrong[answer].mean():.3f}     {answer[~group_b].mean():.3f}       {answer[group_b].mean():.3f}")

d = 0.2                                         # 보류 비용 (오류 비용 1 대비)
reject = conf < 1 - d                           # Chow 규칙
print(f"보류 없음 평균 비용 {wrong.mean():.3f}   Chow 규칙(d = {d}) 평균 비용 {np.mean(np.where(reject, d, wrong)):.3f}"
      f"  보류율 {reject.mean():.3f}")
