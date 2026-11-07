"""21.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/reliability/conformal-prediction/cells.py
"""
# %% cell 1 — split conformal 한 번
import numpy as np

rng = np.random.default_rng(0)
f_true = lambda x: np.sin(2 * x)
sd_true = lambda x: 0.05 + 0.2 * x              # 잡음이 x에 따라 커진다

def sample(n, lo=0.0, hi=3.0):
    x = rng.uniform(lo, hi, n)
    return x, f_true(x) + rng.normal(size=n) * sd_true(x)

x_tr, y_tr = sample(500)                        # 학습용
coef = np.polyfit(x_tr, y_tr, 5)                # 아무 모델이나 된다: 여기서는 5차 다항식
mu = lambda x: np.polyval(coef, x)

alpha = 0.1
x_cal, y_cal = sample(500)                      # 보정용: 학습에 쓰지 않은 데이터
scores = np.abs(y_cal - mu(x_cal))              # (n,) 비순응 점수
n = len(scores)
k = int(np.ceil((n + 1) * (1 - alpha)))         # 보정 점수 중 k번째로 작은 값
q_hat = np.sort(scores)[k - 1]

x_te, y_te = sample(20000)
covered = np.abs(y_te - mu(x_te)) <= q_hat
print(f"n = {n}, k = {k}, q_hat = {q_hat:.3f}")
print(f"시험 포함률 {covered.mean():.3f}  (목표 ≥ {1 - alpha})   구간 폭 {2 * q_hat:.3f}")

# %% cell 2 — 보정 데이터를 바꿔 가며 포함 보장 확인
from scipy.stats import beta, norm

x_grid = rng.uniform(0, 3, 2000)                # 참 포함률을 계산할 x 표본
bias, sd = f_true(x_grid) - mu(x_grid), sd_true(x_grid)
def true_coverage(q):                           # P(|Y - mu(X)| <= q): 시험 데이터가 무한할 때의 포함률
    return np.mean(norm.cdf((q - bias) / sd) - norm.cdf((-q - bias) / sd))

for n in (19, 500):
    k = int(np.ceil((n + 1) * (1 - alpha)))
    cov = np.array([true_coverage(np.sort(np.abs(y - mu(x)))[k - 1]) for x, y in (sample(n) for _ in range(1000))])
    b = beta(k, n + 1 - k)                      # 이론: 포함률 ~ Beta(k, n + 1 - k)
    print(f"n = {n:3d}  평균 {cov.mean():.3f} (이론 {b.mean():.3f})  sd {cov.std():.3f} (이론 {b.std():.3f})"
          f"  0.85 미만인 보정 세트 {np.mean(cov < 0.85):.1%}")

# %% cell 3 — 조건부 포함의 실패와 정규화 점수
r_tr = y_tr - mu(x_tr)
lv = np.polyfit(x_tr, np.log(r_tr ** 2) + 1.27, 2)        # 학습 데이터로 log 분산 회귀 (21.1장)
sig_hat = lambda x: np.exp(0.5 * np.polyval(lv, x))

def split_conformal(score_cal, alpha=0.1):
    k = int(np.ceil((len(score_cal) + 1) * (1 - alpha)))
    return np.sort(score_cal)[k - 1]

q_abs = split_conformal(np.abs(y_cal - mu(x_cal)))                    # 폭이 어디서나 같은 구간
q_norm = split_conformal(np.abs(y_cal - mu(x_cal)) / sig_hat(x_cal))  # 폭이 sig_hat(x)에 비례하는 구간
half = {"절댓값 점수": np.full_like(x_te, q_abs), "정규화 점수": q_norm * sig_hat(x_te)}
bins = [(0, 1), (1, 2), (2, 3)]
print("점수          전체    x∈[0,1) [1,2)  [2,3)   평균 폭")
for name, h in half.items():
    inside = np.abs(y_te - mu(x_te)) <= h
    by_bin = "  ".join(f"{inside[(x_te >= a) & (x_te < b)].mean():.3f}" for a, b in bins)
    print(f"{name}  {inside.mean():.3f}   {by_bin}   {2 * h.mean():.3f}")

# %% cell 4 — 공변량 이동과 가중 conformal
x_sh = 3 * rng.beta(4, 1, 20000)                  # 시험 입력이 잡음 큰 오른쪽으로 쏠린다
y_sh = f_true(x_sh) + rng.normal(size=20000) * sd_true(x_sh)
w = lambda x: 4 * (x / 3) ** 3                    # 가능도 비 p_test(x) / p_train(x) (알고 있다고 가정)

s_cal = np.abs(y_cal - mu(x_cal))
order = np.argsort(s_cal)
s_sorted, cw = s_cal[order], np.cumsum(w(x_cal)[order])   # 정렬한 점수와 누적 가중치
W = cw[-1]
need = (1 - alpha) * (W + w(x_sh))                # 시험점 자신의 가중치는 +inf 점수에 둔다
idx = np.searchsorted(cw, need)                   # 누적 가중치가 처음으로 need 이상이 되는 위치
q_w = np.where(idx < len(s_cal), s_sorted[np.minimum(idx, len(s_cal) - 1)], np.inf)

r_sh = np.abs(y_sh - mu(x_sh))
print(f"이동 없는 시험 포함률          {np.mean(np.abs(y_te - mu(x_te)) <= q_abs):.3f}")
print(f"이동 후, 보통 split conformal  {np.mean(r_sh <= q_abs):.3f}   폭 {2 * q_abs:.3f}")
print(f"이동 후, 가중 conformal        {np.mean(r_sh <= q_w):.3f}   평균 폭 {2 * q_w.mean():.3f}")
