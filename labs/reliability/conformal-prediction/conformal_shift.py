"""21.3 실습: split conformal과 CQR을 구현하고, 분포 이동에서 포함률이 어떻게 무너지는지 본다.

실행: uv run --project labs python labs/reliability/conformal-prediction/conformal_shift.py

데이터: x ~ U(0, 3), y = sin(2x) + ε, ε ~ N(0, (0.05 + 0.2x)²). 학습 1000개, 보정 1000개. 목표 90%.
방법
  split(절댓값)   점 예측(gradient boosting, 제곱손실) + |y - μ|
  CQR            분위수 회귀(gradient boosting, 5%·95%) + max(q_lo - y, y - q_hi)
시험 상황
  이동 없음      x ~ U(0, 3)
  공변량 이동    x ~ 3·Beta(4, 1)  (잡음 큰 오른쪽으로 쏠림, p(y|x)는 그대로)
  조건부 이동    x ~ U(0, 3), 잡음 sd가 1.5배 (p(y|x)가 바뀜: 장비 교체 등)
마지막으로 잡음이 시간에 따라 서서히 커지는 순차 데이터에서 고정 split conformal과
적응적 conformal(ACI, α_{t+1} = α_t + γ(α - err_t))의 누적 포함률을 비교한다.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(0)
ALPHA = 0.1
f = lambda x: np.sin(2 * x)
sd = lambda x: 0.05 + 0.2 * x


def sample(n, kind="none"):
    x = 3 * rng.beta(4, 1, n) if kind == "covariate" else rng.uniform(0, 3, n)
    scale = 1.5 if kind == "conditional" else 1.0
    return x, f(x) + rng.normal(size=n) * sd(x) * scale


def gbr(**kw):
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0, **kw)


def qhat(scores, alpha=ALPHA):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if k < 1:                       # alpha ≥ 1: 빈 구간
        return -np.inf
    return np.sort(scores)[k - 1] if k <= len(scores) else np.inf   # alpha가 너무 작으면 무한 구간


x_tr, y_tr = sample(1000)
x_cal, y_cal = sample(1000)
X = lambda x: x[:, None]
mean_m = gbr().fit(X(x_tr), y_tr)
lo_m = gbr(loss="quantile", alpha=ALPHA / 2).fit(X(x_tr), y_tr)
hi_m = gbr(loss="quantile", alpha=1 - ALPHA / 2).fit(X(x_tr), y_tr)

q_abs = qhat(np.abs(y_cal - mean_m.predict(X(x_cal))))
q_cqr = qhat(np.maximum(lo_m.predict(X(x_cal)) - y_cal, y_cal - hi_m.predict(X(x_cal))))


def intervals(x):
    m, lo, hi = mean_m.predict(X(x)), lo_m.predict(X(x)), hi_m.predict(X(x))
    return {"split(절댓값)": (m - q_abs, m + q_abs), "CQR": (lo - q_cqr, hi + q_cqr), "분위수 회귀만": (lo, hi)}


print(f"목표 포함률 {1 - ALPHA:.2f}.  각 칸: 전체 포함률 / x ∈ [2, 3]의 포함률 / 평균 폭")
for kind, label in (("none", "이동 없음"), ("covariate", "공변량 이동"), ("conditional", "조건부 이동")):
    x, y = sample(20000, kind)
    row = []
    for name, (lo, hi) in intervals(x).items():
        inside = (lo <= y) & (y <= hi)
        row.append(f"{name} {inside.mean():.3f}/{inside[x >= 2].mean():.3f}/{np.mean(hi - lo):.2f}")
    print(f"{label:8s}  " + "   ".join(row))

# 순차 데이터: 잡음 배율이 1에서 2로 서서히 커진다
T = 3000
x_seq = rng.uniform(0, 3, T)
drift = np.linspace(1.0, 2.0, T)
y_seq = f(x_seq) + rng.normal(size=T) * sd(x_seq) * drift
m_seq = mean_m.predict(X(x_seq))
s_cal = np.abs(y_cal - mean_m.predict(X(x_cal)))
r_seq = np.abs(y_seq - m_seq)

fixed_err = r_seq > q_abs
a_t, gamma, aci_err = ALPHA, 0.01, np.zeros(T, bool)
for t in range(T):
    q_t = qhat(s_cal, a_t)
    aci_err[t] = r_seq[t] > q_t
    a_t = a_t + gamma * (ALPHA - aci_err[t])

print("\n순차 데이터의 구간별 포함률 (잡음 배율 1 → 2)")
for a, b in ((0, 1000), (1000, 2000), (2000, 3000)):
    print(f"t ∈ [{a}, {b})  고정 split {1 - fixed_err[a:b].mean():.3f}   ACI {1 - aci_err[a:b].mean():.3f}")
print(f"전체          고정 split {1 - fixed_err.mean():.3f}   ACI {1 - aci_err.mean():.3f}")
