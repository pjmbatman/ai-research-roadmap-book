"""15.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/data/splitting-and-leakage/cells.py
"""
# %% cell 1 — 무작위 분할과 그룹 분할
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)
G, PER = 16, 15                                               # 그룹 16개 × 샘플 15개
groups = np.repeat(np.arange(G), PER)                         # (240,)
grid = np.stack(np.meshgrid(np.arange(4), np.arange(4)), -1).reshape(G, 2)
centers = 1.2 * grid - 1.8 + rng.uniform(-0.25, 0.25, (G, 2))  # (16, 2) 그룹마다 다른 위치
X = centers[groups] + rng.normal(0, 0.33, size=(G * PER, 2))  # (240, 2)
u = rng.normal(0, 1.0, size=G)                                # (16,) 그룹이 공유하는 효과
y = np.sin(0.8 * X[:, 0]) + 0.3 * X[:, 1] + u[groups] + rng.normal(0, 0.15, G * PER)

knn = KNeighborsRegressor(n_neighbors=5)
rmse = "neg_root_mean_squared_error"
s_rand = cross_val_score(knn, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring=rmse)
s_group = cross_val_score(knn, X, y, groups=groups, cv=GroupKFold(4), scoring=rmse)
print(f"무작위 5-fold: RMSE {-s_rand.mean():.2f}")
print(f"그룹 4-fold  : RMSE {-s_group.mean():.2f}")

# %% cell 2 — 겹치는 window: 무작위 분할과 시간 분할
x = np.zeros(3000)
for k in range(1, 3000):                                      # 정상(stationary)인 AR(1) 신호
    x[k] = 0.97 * x[k - 1] + rng.normal(0, 0.25)
W, H = 30, 10                                                 # 입력 30 step, 10 step 뒤를 예측
starts = np.arange(len(x) - W - H)                            # (2960,) stride 1로 겹치는 window
Xw = np.stack([x[s:s + W] for s in starts])                   # (2960, 30)
yw = x[starts + W + H - 1]                                    # (2960,)

def rmse_of(tr, te):                                          # 1-최근접 이웃 예측의 RMSE
    a, b = Xw[te], Xw[tr]                                     # (n_te, 30), (n_tr, 30)
    d2 = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T   # (n_te, n_tr)
    pred = yw[tr][d2.argmin(axis=1)]                          # 가장 가까운 train window의 정답
    return np.sqrt(np.mean((pred - yw[te]) ** 2))

perm = rng.permutation(len(starts))
cut = int(0.8 * len(starts))
print(f"무작위 분할        : RMSE {rmse_of(perm[:cut], perm[cut:]):.2f}")
gap = W + H                                                   # 경계에서 겹치는 window를 버린다
print(f"시간 분할 + gap {gap}: RMSE {rmse_of(starts[:cut - gap], starts[cut:]):.2f}")

# %% cell 3 — 특징 선택의 누수: 잡음에서 성능을 만든다
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

Xn = rng.normal(size=(100, 5000))                             # (100, 5000) 순수한 잡음 특징
yn = rng.integers(0, 2, size=100)                             # (100,) 특징과 무관한 label
cv = StratifiedKFold(5, shuffle=True, random_state=0)

X_sel = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)    # 전체 데이터의 label로 선택
leaky = cross_val_score(LogisticRegression(), X_sel, yn, cv=cv).mean()
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
honest = cross_val_score(pipe, Xn, yn, cv=cv).mean()         # fold마다 train에서만 선택
print(f"선택 후 CV (누수)   : 정확도 {leaky:.2f}")
print(f"pipeline 안에서 선택: 정확도 {honest:.2f}")

# %% cell 4 — test를 보고 고르면 test가 validation이 된다
X_val, y_val = rng.normal(size=(200, 10)), rng.integers(0, 2, 200)   # label은 무작위
X_new, y_new = rng.normal(size=(200, 10)), rng.integers(0, 2, 200)   # 한 번도 안 본 데이터
best_acc, best_w = 0.0, None
for _ in range(500):                                          # 무작위 모델 500개 중 가장 좋은 것
    w = rng.normal(size=10)                                   # (10,)
    acc = np.mean((X_val @ w > 0) == y_val)
    if acc > best_acc:
        best_acc, best_w = acc, w
print(f"고르는 데 쓴 집합의 정확도: {best_acc:.3f}")
print(f"새 데이터에서의 정확도   : {np.mean((X_new @ best_w > 0) == y_new):.3f}")

# %% cell 5 — 사전학습 데이터와 평가 데이터의 중복 점검
import hashlib

AA = list("ACDEFGHIKLMNPQRSTVWY")
pretrain = ["".join(rng.choice(AA, 30)) for _ in range(2000)]        # 사전학습 서열
test = ["".join(rng.choice(AA, 30)) for _ in range(180)]             # 새 서열 180개
test += pretrain[:15]                                                 # 그대로 복사된 15개
test += [" " + s.lower() for s in pretrain[100:105]]                  # 표기만 다른 5개

def key(seq):
    return hashlib.sha256(seq.strip().upper().encode()).hexdigest()  # 정규화 후 해시

exact = len(set(test) & set(pretrain))
normalized = len({key(s) for s in test} & {key(s) for s in pretrain})
print("문자열 그대로 비교한 중복:", exact)
print("정규화 후 해시로 찾은 중복:", normalized)
