"""30.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/battery-ai/battery-foundation-models/cells.py
"""
# %% cell 1 — 분할 단위에 따라 달라지는 시험 오차
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)
X, y, cell, dset = [], [], [], []
for d, (n_cell, log_life) in enumerate([(40, 3.0), (30, 3.0), (30, 2.8)]):   # 가상 데이터셋 A, B, C (C는 다른 프로토콜)
    for _ in range(n_cell):
        z, u, v = rng.normal(size=3)                   # z: 수명 편차, u: 특징의 셀 고유 잡음, v: 수명과 무관한 셀 고유 값
        for n in range(10, 101, 10):                   # 초기 사이클 10, 20, ..., 100에서 계산한 특징 (행 10개)
            X.append([z + 0.7 * u + 0.05 * rng.normal(), v + 0.05 * rng.normal()])
            cell.append(len(y) // 10); dset.append(d)  # 셀 번호, 데이터셋 번호
            y.append(log_life + 0.15 * z)              # 셀 단위 레이블: log10(사이클 수명)
X, y, cell, dset = np.array(X), np.array(y), np.array(cell), np.array(dset)   # (1000, 2) 셀 100개 × 행 10개

def err(train, test):                                  # 사이클 수명의 평균 절대 백분율 오차 (%)
    pred = KNeighborsRegressor(n_neighbors=5).fit(X[train], y[train]).predict(X[test])
    return 100 * np.mean(np.abs(10 ** (pred - y[test]) - 1))

ab = dset < 2
row_test = ab & (rng.random(len(y)) < 0.2)             # (1) A·B의 행(사이클)을 무작위로 20%
cell_test = np.isin(cell, rng.choice(70, 14, replace=False))   # (2) A·B의 셀 20%를 통째로
print(f"행 단위 무작위 분할 (A·B)  수명 오차 {err(ab & ~row_test, row_test):5.1f} %")
print(f"셀 단위 분할 (A·B)         수명 오차 {err(ab & ~cell_test, cell_test):5.1f} %")
print(f"데이터셋 단위 (A·B → C)    수명 오차 {err(ab, dset == 2):5.1f} %")

# %% cell 2 — 사전학습의 전이 효과를 통제된 비교로 재기
D = 8                                                  # 초기 사이클 특징 수
w_src = rng.normal(size=D) / np.sqrt(D)                # 사전학습 데이터(A·B)가 담은 관계
X_src = rng.normal(size=(300, D))                      # (300, D) 사전학습 셀 300개
w_pre = np.linalg.solve(X_src.T @ X_src + np.eye(D), X_src.T @ (X_src @ w_src + 0.1 * rng.normal(size=300)))

def ridge(Xt, yt, w0, lam=3.0):                        # w0를 향해 줄이는 ridge: w0 = 0이면 처음부터, w_pre면 미세조정
    return w0 + np.linalg.solve(Xt.T @ Xt + lam * np.eye(D), Xt.T @ (yt - Xt @ w0))

targets = {"C (같은 화학, 다른 프로토콜)": w_src + 0.3 * rng.normal(size=D) / np.sqrt(D),
           "D (다른 화학)": rng.normal(size=D) / np.sqrt(D)}
for name, w_t in targets.items():
    Xt = rng.normal(size=(40, D)); yt = Xt @ w_t + 0.1 * rng.normal(size=40)    # 목표 데이터셋 셀 40개
    res = []
    for split in range(100):                           # 레이블 셀 6개 / 시험 셀 34개로 100번 나눈다
        idx = np.random.default_rng(split).permutation(40); tr, te = idx[:6], idx[6:]
        rmse = lambda w: np.sqrt(np.mean((Xt[te] @ w - yt[te]) ** 2))
        res.append([rmse(ridge(Xt[tr], yt[tr], np.zeros(D))), rmse(ridge(Xt[tr], yt[tr], w_pre)), rmse(w_pre)])
    R = np.array(res)                                  # (100, 3) 처음부터, 미세조정, zero-shot
    gain = R[:, 0] - R[:, 1]                           # 양수면 사전학습이 도움
    print(f"{name}: RMSE 중앙값 처음부터 {np.median(R[:, 0]):.3f}, 미세조정 {np.median(R[:, 1]):.3f}, "
          f"zero-shot {np.median(R[:, 2]):.3f}")
    print(f"    미세조정이 나은 분할 {np.mean(gain > 0):.0%}, 이득 5–95% 범위 "
          f"[{np.percentile(gain, 5):+.3f}, {np.percentile(gain, 95):+.3f}]")
