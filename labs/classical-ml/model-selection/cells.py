"""16.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/classical-ml/model-selection/cells.py
"""
# %% cell 1 — k-fold와 group k-fold
import warnings
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore", module="threadpoolctl")      # 브라우저 실행 환경의 무해한 경고만 끈다

def make_cells(rng, n_cells, per_cell=20):                     # 셀마다 측정 20개
    g = np.repeat(np.arange(n_cells), per_cell)                # (N,) 셀 번호
    x = rng.uniform(0, 1, (len(g), 3))                          # (N, 3) 운전 조건
    tag = rng.normal(0, 1, (n_cells, 2))[g]                    # 셀을 식별하는 특징(예: 초기 저항) (N, 2)
    offset = rng.normal(0, 0.5, n_cells)[g]                    # 특징으로 설명되지 않는 셀 편차
    y = np.sin(3 * x[:, 0]) + x[:, 1] + offset + rng.normal(0, 0.1, len(g))
    return np.c_[x, tag], y, g

rng = np.random.default_rng(0)
X, y, g = make_cells(rng, n_cells=30)
X_new, y_new, _ = make_cells(rng, n_cells=200)                 # 처음 보는 셀 200개 (독립 데이터)
group_folds = [(np.flatnonzero(g % 5 != k), np.flatnonzero(g % 5 == k)) for k in range(5)]   # 셀 단위 5-fold
for name, cols in [("조건 + 셀 식별 특징", slice(None)), ("조건만", slice(0, 3))]:
    knn = KNeighborsRegressor(n_neighbors=3)
    r2_kf = cross_val_score(knn, X[:, cols], y, cv=KFold(5, shuffle=True, random_state=0)).mean()
    r2_gkf = cross_val_score(knn, X[:, cols], y, cv=group_folds).mean()
    r2_new = knn.fit(X[:, cols], y).score(X_new[:, cols], y_new)
    print(f"{name:12s} R²  k-fold {r2_kf:5.2f}   group k-fold {r2_gkf:5.2f}   새 셀 {r2_new:5.2f}")

# %% cell 2 — 회귀 metric: 단위와 범위
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

temp_c = np.array([0.2, 1.0, 5.0, 12.0, 20.0, 30.0])           # 참 온도 [°C]
pred_c = temp_c + 0.5                                          # 모든 점에서 0.5 °C 높게 예측
for unit, yt, yp in [("°C", temp_c, pred_c), ("K", temp_c + 273.15, pred_c + 273.15)]:
    print(f"단위 {unit:2s}  MAE {mean_absolute_error(yt, yp):.2f}   MAPE {100 * mean_absolute_percentage_error(yt, yp):6.2f}%")

narrow_c = np.linspace(19, 21, 50)                             # 좁은 범위의 참값
wide_c = np.linspace(0, 40, 50)                                # 넓은 범위의 참값
noise = np.where(np.arange(50) % 2 == 0, 0.5, -0.5)            # 같은 크기(±0.5 °C)의 오차
for name, yt in [("19–21 °C", narrow_c), ("0–40 °C", wide_c)]:
    print(f"범위 {name:9s} MAE {mean_absolute_error(yt, yt + noise):.2f}   R² {r2_score(yt, yt + noise):.3f}")

# %% cell 3 — 불균형 분류의 metric
from scipy.stats import norm
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

def make_scores(rng, n, prior):                                # 양성 비율 prior, 점수 분포는 고정
    yy = (rng.uniform(size=n) < prior).astype(int)             # (n,)
    s = rng.normal(0, 1, n) + 2.0 * yy                         # 양성은 평균 2만큼 높다
    lr = norm.pdf(s, 2, 1) / norm.pdf(s, 0, 1)                 # 우도비
    p = prior * lr / (prior * lr + 1 - prior)                  # 보정된(calibrated) 양성 확률
    return yy, s, p

for prior in (0.5, 0.01):
    yy, s, p = make_scores(rng, 20000, prior)
    pred = (p >= 0.5).astype(int)
    print(f"양성 {prior:4.0%}  accuracy: 전부 음성 {accuracy_score(yy, 0 * yy):.3f}, 분류기 {accuracy_score(yy, pred):.3f}")
    print(f"           precision {precision_score(yy, pred):.2f}  recall {recall_score(yy, pred):.2f}  F1 {f1_score(yy, pred):.2f}"
          f"   ROC-AUC {roc_auc_score(yy, s):.3f}  PR-AUC {average_precision_score(yy, s):.3f}")

# %% cell 4 — 비용으로 threshold 고르기
c_fn, c_fp = 50.0, 1.0                                         # 불량을 놓친 비용, 정상을 재검사한 비용
y_val, _, p_val = make_scores(rng, 20000, 0.01)                # 검증 데이터
y_te, _, p_te = make_scores(rng, 20000, 0.01)                  # 시험 데이터

def cost_per_1000(yy, p, thr):
    pred = p >= thr
    return 1000 * (c_fn * np.sum(~pred & (yy == 1)) + c_fp * np.sum(pred & (yy == 0))) / len(yy)

grid = np.linspace(0.001, 0.5, 500)
thr_val = grid[np.argmin([cost_per_1000(y_val, p_val, t) for t in grid])]
for name, thr in [("0.5 (기본값)", 0.5), ("이론값 c_fp/(c_fp+c_fn)", c_fp / (c_fp + c_fn)), ("검증 데이터에서 고름", thr_val)]:
    print(f"{name:24s} threshold {thr:.3f}   시험 데이터 비용 {cost_per_1000(y_te, p_te, thr):6.1f} / 1000개")

# %% cell 5 — 탐색 점수의 낙관 편향과 nested CV
def feature_acc(Xtr, ytr, Xte, yte):                           # 특징 하나씩 쓰는 최근접 평균 분류기들
    m0, m1 = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)      # (D,), (D,)
    pred = np.abs(Xte - m1) < np.abs(Xte - m0)                 # (n_te, D) 설정(특징)마다 예측
    return (pred == yte[:, None]).mean(0)                      # (D,) 설정마다 정확도

def cv_acc(Xd, yd, k=5):                                       # 설정마다 k-fold CV 정확도, (D,)
    fold = np.arange(len(yd)) % k
    return np.mean([feature_acc(Xd[fold != f], yd[fold != f], Xd[fold == f], yd[fold == f])
                    for f in range(k)], axis=0)

flat, nested = [], []
for d in range(200):                                           # label이 특징과 무관한 데이터셋 200개
    r = np.random.default_rng(100 + d)
    Xn, yn = r.normal(size=(40, 50)), r.permutation(np.repeat([0, 1], 20))   # (40, 50), 참 정확도 0.5
    flat.append(cv_acc(Xn, yn).max())                          # 설정 50개 중 CV 점수 최고값
    outer = np.arange(40) % 5
    nested.append(np.mean([feature_acc(Xn[outer != f], yn[outer != f], Xn[outer == f], yn[outer == f])[
        np.argmax(cv_acc(Xn[outer != f], yn[outer != f]))]    # 안쪽 CV로 고른 설정을 바깥 fold에서 평가
        for f in range(5)]))
print(f"탐색에 쓴 CV 최고 점수: 평균 {np.mean(flat):.3f}   0.6 이상인 데이터셋 {np.mean(np.array(flat) >= 0.6):.0%}")
print(f"nested CV 추정값:       평균 {np.mean(nested):.3f}   0.6 이상인 데이터셋 {np.mean(np.array(nested) >= 0.6):.0%}")
