"""1.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-landscape/modeling-axes/cells.py
"""
# %% cell 1 — 같은 데이터, 판별 모델과 생성 모델
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
cov = np.array([[1.0, 0.3], [0.3, 1.0]])
X0 = rng.multivariate_normal([0.0, 0.0], cov, size=300)    # (300, 2) 정상 시료의 두 측정값
X1 = rng.multivariate_normal([2.0, 1.0], cov, size=300)    # (300, 2) 불량 시료
X, y = np.vstack([X0, X1]), np.repeat([0, 1], 300)         # (600, 2), (600,)

disc = LogisticRegression().fit(X, y)                      # p(y|x)를 직접 학습
gen = LinearDiscriminantAnalysis(store_covariance=True).fit(X, y)   # p(x|y)p(y)를 학습

x_new = np.array([[1.0, 0.5], [12.0, 6.0]])                # 경계 근처 점, 데이터에서 먼 점
print("판별 p(y=1|x):", disc.predict_proba(x_new)[:, 1].round(3))
print("생성 p(y=1|x):", gen.predict_proba(x_new)[:, 1].round(3))
p_x = sum(gen.priors_[k] * multivariate_normal(gen.means_[k], gen.covariance_).pdf(x_new)
          for k in (0, 1))                                 # p(x) = sum_k p(x|y=k) p(y=k)
print("생성 p(x)    :", np.array2string(p_x, precision=2))
print("생성모델이 만든 불량 시료 3개:\n", rng.multivariate_normal(gen.means_[1], gen.covariance_, size=3).round(2))

# %% cell 2 — 판별 모델이 출력하는 예측분포
from sklearn.linear_model import LinearRegression

dtemp_k = rng.uniform(0, 10, size=200)                     # (200,) 입력: 온도 차
volt_mv = 0.5 * dtemp_k + rng.normal(0, 0.4, size=200)     # (200,) 출력: 열전 전압
reg = LinearRegression().fit(dtemp_k[:, None], volt_mv)
sigma_mv = np.std(volt_mv - reg.predict(dtemp_k[:, None]), ddof=2)   # 잔차로 추정한 잡음 표준편차

new_k = rng.uniform(0, 10, size=5000)                      # (5000,) 새 입력
new_mv = 0.5 * new_k + rng.normal(0, 0.4, size=5000)       # (5000,)
mu_mv = reg.predict(new_k[:, None])                        # (5000,) 점 예측 = 예측분포의 평균
inside = np.abs(new_mv - mu_mv) <= 1.645 * sigma_mv        # 90% 예측 구간 mu ± 1.645 sigma
f5 = reg.predict([[5.0]])[0]
print(f"점 예측    f(5 K) = {f5:.3f} mV")
print(f"예측분포 p(y|5 K) = N({f5:.3f}, {sigma_mv:.3f}^2) mV")
print(f"90% 예측 구간의 실제 포함률 {inside.mean():.3f}")

# %% cell 3 — 파라미터를 보는 두 관점
from scipy.stats import beta

for k, n in [(3, 3), (240, 300)]:                          # 성공 k번 / 시도 n번
    mle = k / n                                            # 빈도주의: 고정된 미지수의 점 추정
    post = beta(1 + k, 1 + n - k)                          # 베이지안: 균등 사전분포 Beta(1, 1)의 사후분포
    lo, hi = post.ppf([0.025, 0.975])
    print(f"{k:3d}/{n:<3d}  MLE {mle:.3f}   사후 평균 {post.mean():.3f}   95% 구간 [{lo:.3f}, {hi:.3f}]")
