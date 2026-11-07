"""12.5 실습: 두 방법을 5개 seed와 셀 단위 그룹 분할로 비교하고, 차이를 주장할 수 있는지 판단한다.

데이터와 방법은 본문(cells.py)과 같다. 셀 30개 × 측정 40개, 방법 A = 무작위 특징 30개, B = 무작위 특징 100개.
seed마다 (1) 셀을 5개 fold로 나누는 방식과 (2) 무작위 특징이 함께 바뀐다.

세 가지 질문에 차례로 답한다.
  1. seed마다 결과가 얼마나 흔들리는가 (평균 ± 표준편차, 표준오차)
  2. 같은 seed·같은 fold에서 짝지은 차이는 얼마이고 seed 사이에 일관적인가
  3. 이 차이가 '새 셀'에도 일반화되는가 (셀 단위 짝지은 부트스트랩)

실행: uv run --project labs python labs/statistical-inference/comparing-models/compare_methods.py
"""
import numpy as np

T975_DF4 = 2.776   # t 분포 97.5% 분위수, 자유도 4 (seed 5개)


def make_data(rng, n_cells=30, per_cell=40):
    x = rng.uniform(-2, 2, size=(n_cells * per_cell, 2))
    cell = np.repeat(np.arange(n_cells), per_cell)
    offset = rng.normal(0, 0.3, n_cells)[cell]
    y = np.sin(2.0 * x[:, 0]) * np.cos(x[:, 1]) + offset + rng.normal(0, 0.2, len(x))
    return x, y, cell


def fit_predict(x_tr, y_tr, x_te, seed, n_feat, scale=1.5):
    r = np.random.default_rng(seed)
    W, b = r.normal(0, scale, (2, n_feat)), r.uniform(0, 2 * np.pi, n_feat)
    P = np.cos(x_tr @ W + b)
    w = np.linalg.solve(P.T @ P + 1e-3 * len(y_tr) * np.eye(n_feat), P.T @ y_tr)
    return np.cos(x_te @ W + b) @ w


def oof_sq_errors(x, y, cell, seed, n_feat, k=5):
    """셀 단위 k-fold로 모든 샘플의 out-of-fold 제곱오차를 구한다."""
    fold = np.random.default_rng(seed).permutation(np.unique(cell)) % k   # 셀마다 fold 번호
    err = np.empty(len(y))
    for j in range(k):
        te = fold[cell] == j
        err[te] = (fit_predict(x[~te], y[~te], x[te], seed, n_feat) - y[te]) ** 2
    return err


def main():
    x, y, cell = make_data(np.random.default_rng(0))
    seeds = range(5)
    ea = np.array([oof_sq_errors(x, y, cell, s, 30) for s in seeds])     # (S, N)
    eb = np.array([oof_sq_errors(x, y, cell, s, 100) for s in seeds])    # (S, N)
    ra, rb = np.sqrt(ea.mean(axis=1)), np.sqrt(eb.mean(axis=1))           # seed별 CV RMSE, (S,)
    d = rb - ra

    print("seed   A RMSE   B RMSE   B - A")
    for s in seeds:
        print(f"{s:4d}   {ra[s]:.4f}   {rb[s]:.4f}   {d[s]:+.4f}")

    print("\n1. 방법별 요약 (seed 5개)")
    for name, r in (("A", ra), ("B", rb)):
        sd = r.std(ddof=1)
        print(f"   {name}: 평균 {r.mean():.4f}   표준편차 {sd:.4f}   표준오차 {sd / np.sqrt(len(r)):.4f}")

    se_d = d.std(ddof=1) / np.sqrt(len(d))
    print("\n2. 짝지은 차이 B - A (같은 seed, 같은 fold)")
    print(f"   평균 {d.mean():+.4f}   표준편차 {d.std(ddof=1):.4f}   95% 구간 (t, 자유도 4) "
          f"[{d.mean() - T975_DF4 * se_d:+.4f}, {d.mean() + T975_DF4 * se_d:+.4f}]   B가 이긴 seed {np.sum(d < 0)}/5")
    print(f"   짝을 짓지 않았다면 차이의 표준오차 {np.sqrt(ra.var(ddof=1) / 5 + rb.var(ddof=1) / 5):.4f} (짝지은 경우 {se_d:.4f})")

    cells = np.unique(cell)
    cell_diff = np.array([(eb[:, cell == c] - ea[:, cell == c]).mean() for c in cells])   # 셀별 MSE 차이, (C,)
    rng = np.random.default_rng(1)
    boot = cell_diff[rng.integers(0, len(cells), size=(5000, len(cells)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("\n3. 새 셀로의 일반화: 셀 30개를 재표본화한 짝지은 부트스트랩 (MSE 차이, seed 평균)")
    print(f"   MSE 차이 {cell_diff.mean():+.5f}   95% 구간 [{lo:+.5f}, {hi:+.5f}]   B가 나은 셀 {np.sum(cell_diff < 0)}/{len(cells)}")


if __name__ == "__main__":
    main()
