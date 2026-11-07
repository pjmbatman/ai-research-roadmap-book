"""11.3 실습: 기저율에 따른 양성 예측도와, 재검사에서 조건부 독립 가정이 깨질 때의 영향.

실행: uv run --project labs python labs/probability/conditional-probability/base_rate_table.py
"""
import numpy as np

SENS, FPR = 0.99, 0.05            # 민감도 P(+|병), 위양성률 P(+|정상)
PERSIST = 0.5                     # 정상인 위양성 중 '사람 특성' 때문이라 재검사에서도 양성이 나오는 비율
N = 4_000_000
rng = np.random.default_rng(0)


def bayes(prior, lik1, lik0):
    return lik1 * prior / (lik1 * prior + lik0 * (1 - prior))


print(f"sens {SENS}, FPR {FPR}, persist {PERSIST}, N = {N:,}, seed 0")
print("P(sick | positive): formula vs simulation")
print(f"{'prev':>8} | {'1 test':>9} {'sim':>7} | {'2 tests indep':>18} {'sim indep':>10} {'sim persist':>16}")
for prev in (0.001, 0.01, 0.1, 0.5):
    sick = rng.random(N) < prev
    t1 = np.where(sick, rng.random(N) < SENS, rng.random(N) < FPR)
    # 재검사 (a) 조건부 독립: 첫 검사와 무관하게 다시 뽑는다
    t2_ind = np.where(sick, rng.random(N) < SENS, rng.random(N) < FPR)
    # 재검사 (b) 정상인의 위양성 중 PERSIST 비율은 사람 특성이라 재검사에서도 양성
    stuck = t1 & ~sick & (rng.random(N) < PERSIST)
    t2_cor = np.where(stuck, True, t2_ind)
    print(f"{prev:8.3f} | {bayes(prev, SENS, FPR):9.4f} {sick[t1].mean():7.4f} | "
          f"{bayes(prev, SENS**2, FPR**2):18.4f} {sick[t1 & t2_ind].mean():10.4f} {sick[t1 & t2_cor].mean():16.4f}")
