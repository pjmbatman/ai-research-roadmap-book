"""16.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/classical-ml/trees-and-ensembles/cells.py
"""
# %% cell 1 — 결정 트리: 깊이와 과적합
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def make_alloy(n, rng):                                    # 가상의 합금 시편 표 데이터
    temp_c = rng.uniform(450, 650, n)                      # 시효 온도 [°C]
    time_h = rng.uniform(0.5, 12, n)                       # 시효 시간 [h]
    cu = rng.uniform(0, 0.05, n)                           # Cu 질량분율
    mg = rng.uniform(0, 0.02, n)                           # Mg 질량분율
    line = rng.integers(0, 5, n)                           # 공정 라인 (범주형, 코드 0–4)
    noise_x = rng.normal(size=n)                           # 출력과 무관한 연속 특징
    lot_id = rng.integers(0, 1000, n)                      # 출력과 무관한 로트 번호(값의 종류가 많다)
    peak = (temp_c > 520) & (temp_c < 600) & (time_h > 3)  # 최적 시효 조건에서 강도가 뛴다
    y = (200 + 3000 * cu + 5000 * mg + 40 * peak + 15 * np.sin(time_h)
         + np.array([0, 10, -5, 15, 0])[line] + rng.normal(0, 10, n))   # 항복강도 [MPa]
    X = np.column_stack([temp_c, time_h, cu, mg, line, noise_x, lot_id])  # (n, 7)
    return X, y

names = ["temp_c", "time_h", "cu", "mg", "line", "noise_x", "lot_id"]
rng = np.random.default_rng(0)
X_tr, y_tr = make_alloy(1000, rng)
X_te, y_te = make_alloy(1000, rng)
rmse = lambda m, X, y: np.sqrt(np.mean((m.predict(X) - y) ** 2))
for depth in (1, 3, 6, 10, None):
    tree = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    print(f"깊이 {str(depth):>4s}: 잎 {tree.get_n_leaves():4d}개  학습 RMSE {rmse(tree, X_tr, y_tr):5.1f}  시험 RMSE {rmse(tree, X_te, y_te):5.1f} MPa")

# %% cell 2 — bagging과 랜덤 포레스트, OOB 오차
from sklearn.ensemble import RandomForestRegressor

for label, mf in [("bagging (특징 전부)", 1.0), ("랜덤 포레스트 (특징 1/3)", 0.33)]:
    rf = RandomForestRegressor(n_estimators=300, max_features=mf, oob_score=True, random_state=0, n_jobs=1)
    rf.fit(X_tr, y_tr)
    oob_rmse = np.sqrt(np.mean((rf.oob_prediction_ - y_tr) ** 2))
    print(f"{label}: OOB RMSE {oob_rmse:5.1f}  시험 RMSE {rmse(rf, X_te, y_te):5.1f} MPa")

# %% cell 3 — gradient boosting = 잔차에 트리를 맞추는 반복
lr = 0.1                                                   # 학습률(한 걸음의 크기)
F = np.full(len(y_tr), y_tr.mean())                        # 초기 예측: 평균, (N,)
F_te = np.full(len(y_te), y_tr.mean())
for m in range(1, 301):
    h = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_tr, y_tr - F)   # 잔차 = 음의 gradient
    F += lr * h.predict(X_tr)
    F_te += lr * h.predict(X_te)
    if m in (1, 10, 100, 300):
        print(f"단계 {m:3d}: 학습 RMSE {np.sqrt(np.mean((F - y_tr) ** 2)):3.0f}  시험 RMSE {np.sqrt(np.mean((F_te - y_te) ** 2)):3.0f} MPa")

# %% cell 4 — 표 데이터의 강한 baseline 비교
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

models = {
    "Ridge (16.2장)": make_pipeline(StandardScaler(), RidgeCV()),
    "랜덤 포레스트": RandomForestRegressor(n_estimators=200, max_features=0.33, random_state=0, n_jobs=1),
    "gradient boosting": GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05, max_depth=3,
                                                   n_iter_no_change=20, validation_fraction=0.2, random_state=0),
}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    print(f"시험 RMSE {rmse(model, X_te, y_te):3.0f} MPa  {name}")
print("early stopping이 트리 2000개 중 500개 전에 멈췄는가:", models["gradient boosting"].n_estimators_ < 500)

# %% cell 5 — 특징 중요도의 함정
from sklearn.inspection import permutation_importance

rf = models["랜덤 포레스트"]
perm = permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=0, n_jobs=1)
print("특징      impurity 중요도   permutation 중요도(시험 R² 감소)")
for j, name in enumerate(names):
    print(f"{name:8s}  {rf.feature_importances_[j]:.2f}              {round(perm.importances_mean[j], 2) + 0.0:+.2f}")
