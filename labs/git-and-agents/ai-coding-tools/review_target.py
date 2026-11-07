"""(4.3 실습용) AI 코딩 도구가 작성했다고 가정한 학습 스크립트. **결함이 여러 개 숨어 있다.**

요청한 과제: 배터리 셀 8개의 충방전 기록으로 방전 용량(Ah)을 예측하는 선형 모델을 학습하고,
**처음 보는 셀**에 대한 예측 오차(RMSE, Ah)를 보고한다. L2 정규화 세기 lam은 검증으로 고른다.

이 파일은 돌아가고, 그럴듯한 숫자를 출력한다. shape·gradient·데이터 분할·metric 관점에서 리뷰한다.
실행: uv run --project labs python labs/git-and-agents/ai-coding-tools/review_target.py
해설은 같은 폴더의 review_key.md에 있다(리뷰를 마친 뒤에 연다).
"""
import numpy as np


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    n_cells, n_cycles = 8, 60
    cell = np.repeat(np.arange(n_cells), n_cycles)                    # (N,)
    cycle = np.tile(np.arange(n_cycles), n_cells).astype(float)       # (N,)
    temp_c = (25 + 10 * rng.random(n_cells))[cell]                    # (N,) 셀마다 다른 시험 온도
    cap0_ah = (2.0 + 0.05 * rng.normal(size=n_cells))[cell]           # (N,) 셀마다 다른 초기 용량
    capacity_ah = cap0_ah - 0.002 * cycle * temp_c / 25 + 0.005 * rng.normal(size=cell.size)
    return cycle, temp_c, cell, capacity_ah[:, None]                  # y: (N, 1)


def features(cycle, temp_c, cell, n_cells=8):
    onehot = np.eye(n_cells)[cell]                                    # (N, n_cells) 셀 차이를 반영
    return np.column_stack([cycle / 60, temp_c / 35, cycle * temp_c / 2100, onehot])   # (N, 3 + n_cells)


def mse_loss(w, X, y, lam):
    pred = X @ w                                                      # (N,)
    return np.mean((pred - y) ** 2) + lam * np.sum(w ** 2)


def grad(w, X, y, lam):
    pred = X @ w                                                      # (N,)
    return 2 * X.T @ (pred - y[:, 0]) / len(X) - 2 * lam * w          # (D,)


def fit(X, y, lam, lr=0.1, steps=3000):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w -= lr * grad(w, X, y, lam)
    return w


def rmse(pred, y):
    return np.mean(np.abs(pred - y[:, 0]))


def main():
    cycle, temp_c, cell, y = make_data()
    X = features(cycle, temp_c, cell)
    idx = np.random.default_rng(1).permutation(len(X))
    train, test = idx[:360], idx[360:]
    results = {}
    for lam in [0.0, 1e-4, 1e-3, 1e-2]:
        w = fit(X[train], y[train], lam)
        results[lam] = rmse(X[test] @ w, y[test])
        print(f"lam={lam:<6g} train loss={mse_loss(w, X[train], y[train], lam):.4f}  test RMSE={results[lam]:.4f} Ah")
    best = min(results, key=results.get)
    print(f"최종: lam={best:g}, 처음 보는 셀에 대한 RMSE = {results[best]:.4f} Ah")


if __name__ == "__main__":
    main()
