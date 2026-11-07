"""16.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/classical-ml/linear-models/cells.py
"""
# %% cell 1 — basis 확장: 선형 모델로 곡선 맞추기
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

rng = np.random.default_rng(0)
f = lambda t: np.exp(-t) * np.cos(4 * t)                   # 감쇠 진동 (참 함수)
t_tr = np.sort(rng.uniform(0, 2, 25))[:, None]             # 학습 입력 [s], (25, 1)
y_tr = f(t_tr[:, 0]) + rng.normal(0, 0.05, 25)
t_in = np.linspace(0, 2, 200)[:, None]                     # 학습 구간 안, (200, 1)
t_out = np.linspace(2, 2.5, 50)[:, None]                   # 학습 구간 밖, (50, 1)
rmse = lambda m, t: np.sqrt(np.mean((m.predict(t) - f(t[:, 0])) ** 2))

for name, feat in [("원 특징 t", PolynomialFeatures(1)), ("다항 5차", PolynomialFeatures(5)),
                   ("다항 12차", PolynomialFeatures(12)), ("스플라인", SplineTransformer(n_knots=11))]:
    m = make_pipeline(StandardScaler(), feat, LinearRegression()).fit(t_tr, y_tr)
    print(f"특징 {m[-1].coef_.size:2d}개  구간 안 RMSE {rmse(m, t_in):.3f}  구간 밖 RMSE {rmse(m, t_out):7.2f}  {name}")

# %% cell 2 — Lasso의 정규화 경로
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, Ridge

data = load_diabetes()
names = np.array(data.feature_names)
X = (data.data - data.data.mean(0)) / data.data.std(0)     # 표준화한 특징, (442, 10)
y = data.target                                            # 1년 뒤 질병 진행 지표, (442,)
for alpha in [40.0, 20.0, 10.0, 3.0, 0.3]:
    w_l = Lasso(alpha=alpha).fit(X, y).coef_               # (10,)
    w_r = Ridge(alpha=2 * len(y) * alpha).fit(X, y).coef_  # 같은 척도의 L2 벌점
    print(f"alpha {alpha:4.1f}  0 아닌 계수 Lasso {np.sum(w_l != 0):2d}개, Ridge {np.sum(w_r != 0):2d}개  "
          f"Lasso 선택: {', '.join(names[w_l != 0])}")

# %% cell 3 — 단위를 바꾸면 Lasso의 선택이 바뀐다
raw = load_diabetes(scaled=False).data                     # 원래 단위(나이 [년], 혈압 [mmHg] ...), (442, 10)
raw_kpa = raw.copy()
raw_kpa[:, 3] *= 0.1333                                    # 혈압 bp: mmHg → kPa
for label, Xr in [("원래 단위", raw), ("bp를 kPa로", raw_kpa)]:
    w = Lasso(alpha=10.0).fit(Xr, y).coef_
    pipe = make_pipeline(StandardScaler(), Lasso(alpha=10.0)).fit(Xr, y)
    print(f"[{label}] 표준화 없이: {', '.join(names[w != 0])} / 표준화 후: {', '.join(names[pipe[-1].coef_ != 0])}")

# %% cell 4 — 교차검증으로 강도 고르기, 그리고 선택의 안정성
from sklearn.linear_model import LassoCV

cv_model = make_pipeline(StandardScaler(), LassoCV(cv=5)).fit(raw, y)
alpha_cv = cv_model[-1].alpha_
print(f"CV로 고른 alpha = {alpha_cv:.2f}, 선택된 특징: {', '.join(names[cv_model[-1].coef_ != 0])}")
boot_rng = np.random.default_rng(0)
freq = np.zeros(10)
for _ in range(200):                                       # 부트스트랩 표본마다 같은 alpha로 다시 적합
    i = boot_rng.integers(0, len(y), len(y))
    freq += make_pipeline(StandardScaler(), Lasso(alpha=alpha_cv)).fit(raw[i], y[i])[-1].coef_ != 0
print("부트스트랩 200번 중 선택 비율:", "  ".join(f"{n} {p:.2f}" for n, p in zip(names, freq / 200)))

# %% cell 5 — 로지스틱 회귀: 확률, 결정 경계, threshold
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(3)
stress_mpa = rng.uniform(200, 400, 600)                    # 응력 진폭 [MPa], (600,)
temp_c = rng.uniform(20, 120, 600)                         # 시험 온도 [°C], (600,)
logit = 0.05 * (stress_mpa - 330) + 0.03 * (temp_c - 70)   # 파손의 log-odds (참 모형)
failed = rng.random(600) < 1 / (1 + np.exp(-logit))        # 파손 여부, (600,)
Xf = np.column_stack([stress_mpa, temp_c])                 # (600, 2)
clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xf[:400], failed[:400])
p = clf.predict_proba(Xf[400:])[:, 1]                      # 시험 시편의 파손 확률, (200,)
yt = failed[400:]
for thr in (0.5, 0.2):
    alarm = p >= thr
    print(f"threshold {thr}: 실제 파손 중 경보 비율 {alarm[yt].mean():.2f}, 경보 중 실제 파손 비율 {yt[alarm].mean():.2f}")
w = clf[-1].coef_[0] / clf[0].scale_                       # 원래 단위의 계수 [1/MPa, 1/°C]
b = clf[-1].intercept_[0] - w @ clf[0].mean_
print(f"70 °C에서 결정 경계(p = 0.5) 응력 = {-(b + w[1] * 70) / w[0]:.0f} MPa")
