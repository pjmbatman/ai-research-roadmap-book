"""25.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/scientific-discovery/scientific-foundation-models/cells.py
"""
# %% cell 1 — 통제된 전이 비교: 사전학습 표현 대 원시 입력, 데이터 크기별
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

rng = np.random.default_rng(1)
D_obs, D_lat = 60, 4                                  # 관측 60차원, 잠재 요인 4개
A_task, A_other = rng.normal(size=(D_lat, D_obs)), rng.normal(size=(D_lat, D_obs))
v = rng.normal(size=D_lat)
def sample(n, A):                                     # x = z A + 잡음, 성질 y = z v + 잡음
    z = rng.normal(size=(n, D_lat))
    return z @ A + 2.0 * rng.normal(size=(n, D_obs)), z @ v + 0.1 * rng.normal(size=n)

pre_same = PCA(D_lat).fit(sample(5000, A_task)[0])    # 같은 생성 과정의 비표지 데이터로 "사전학습"
pre_other = PCA(D_lat).fit(sample(5000, A_other)[0])  # 다른 생성 과정의 데이터로 "사전학습"
X_te, y_te = sample(2000, A_task)
alphas = np.logspace(-3, 3, 13)
for n in [10, 30, 100, 1000]:
    scores = {"원시 입력": [], "같은 분포 사전학습": [], "다른 분포 사전학습": []}
    for _ in range(20):                               # 표지 데이터를 20번 새로 뽑는다
        X_tr, y_tr = sample(n, A_task)
        for name, f in [("원시 입력", lambda X: X), ("같은 분포 사전학습", pre_same.transform),
                        ("다른 분포 사전학습", pre_other.transform)]:
            scores[name].append(RidgeCV(alphas=alphas).fit(f(X_tr), y_tr).score(f(X_te), y_te))
    print(f"표지 {n:4d}개  " + "  ".join(f"{k} R² {np.median(s):6.3f}" for k, s in scores.items()))

# %% cell 2 — 평가 누수: 비슷한 시료가 학습과 시험에 갈라져 들어가면
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)
n_family, n_member, D = 80, 8, 8                      # 골격(계열) 80개, 계열마다 변형 8개, 특징 8차원
centers = rng.normal(size=(n_family, D))              # (80, D) 계열의 공통 골격
X = np.repeat(centers, n_member, 0) + 0.1 * rng.normal(size=(n_family * n_member, D))   # (640, D)
family = np.repeat(np.arange(n_family), n_member)     # (640,) 계열 번호
y = np.tanh(centers @ rng.normal(size=D) / 2)[family] + 0.1 * rng.normal(size=len(X))   # 성질은 주로 골격이 정한다

def evaluate(test_mask):
    tr, te = ~test_mask, test_mask
    pred = KNeighborsRegressor(5).fit(X[tr], y[tr]).predict(X[te])
    r2 = 1 - np.mean((pred - y[te]) ** 2) / np.var(y[te])
    nn = np.min(np.linalg.norm(X[te][:, None] - X[tr][None], axis=-1), axis=1)   # 가장 가까운 학습 시료까지 거리
    return r2, np.median(nn)

random_test = rng.random(len(X)) < 0.25                             # 시료 단위 무작위 분할
family_test = np.isin(family, rng.choice(n_family, 20, replace=False))   # 계열 단위 분할
for name, m in [("무작위 분할", random_test), ("계열 단위 분할", family_test)]:
    r2, nn = evaluate(m)
    print(f"{name:8s} 시험 시료 {m.sum()}개  R² = {r2:.3f}   가장 가까운 학습 시료까지 거리 중앙값 {nn:.2f}")
