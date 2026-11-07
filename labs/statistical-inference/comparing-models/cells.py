"""12.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/statistical-inference/comparing-models/cells.py
"""
# %% cell 1 — 데이터와 두 방법
import numpy as np

def make_data(rng, n_cells=30, per_cell=40, freq=2.0):   # 셀 30개 × 측정 40개
    x = rng.uniform(-2, 2, size=(n_cells * per_cell, 2))                  # (N, 2)
    cell = np.repeat(np.arange(n_cells), per_cell)                         # (N,)
    offset = rng.normal(0, 0.3, n_cells)[cell]                             # 셀마다 다른 편차
    y = np.sin(freq * x[:, 0]) * np.cos(x[:, 1]) + offset + rng.normal(0, 0.2, len(x))
    return x, y, cell

def fit_predict(x_tr, y_tr, x_te, seed, n_feat, scale=1.5):   # 무작위 특징 ridge 회귀
    r = np.random.default_rng(seed)                          # 학습 seed가 특징을 정한다
    W, b = r.normal(0, scale, (2, n_feat)), r.uniform(0, 2 * np.pi, n_feat)
    P = np.cos(x_tr @ W + b)                                 # (N_tr, F)
    w = np.linalg.solve(P.T @ P + 1e-3 * len(y_tr) * np.eye(n_feat), P.T @ y_tr)
    return np.cos(x_te @ W + b) @ w

def group_split(cell, split_seed, test_frac=0.3):          # 셀 단위 분할
    cells = np.random.default_rng(split_seed).permutation(np.unique(cell))
    test = np.isin(cell, cells[: int(len(cells) * test_frac)])
    return ~test, test

x, y, cell = make_data(np.random.default_rng(0))
def rmse(n_feat, seed, split):
    tr, te = group_split(cell, split)
    return np.sqrt(np.mean((fit_predict(x[tr], y[tr], x[te], seed, n_feat) - y[te]) ** 2))

print(f"데이터 {len(y)}개, 셀 {cell.max() + 1}개")
print(f"seed 0, 분할 0:  A(특징 30개) RMSE {rmse(30, 0, 0):.4f}   B(특징 100개) RMSE {rmse(100, 0, 0):.4f}")

# %% cell 2 — 변동의 출처: seed와 데이터 분할
S = 10
R = {f: np.array([[rmse(f, s, sp) for s in range(S)] for sp in range(S)]) for f in (30, 100)}   # (분할, seed)
for f, name in ((30, "A"), (100, "B")):
    r = R[f]
    print(f"{name}: 평균 {r.mean():.4f}   seed만 바꿀 때 sd {r.std(axis=1, ddof=1).mean():.4f}"
          f"   분할만 바꿀 때 sd {r.std(axis=0, ddof=1).mean():.4f}")

D = R[100] - R[30]                                        # 같은 (분할, seed)에서의 차이 B - A
print(f"\n짝지은 차이 B - A: 평균 {D.mean():+.4f}, sd {D.std(ddof=1):.4f}   B가 이긴 조합 {np.mean(D < 0):.0%}")
print("분할 0에서 seed별 차이:", " ".join(f"{v:+.4f}" for v in D[0]))

# %% cell 3 — 같은 테스트 샘플에서의 짝지은 비교
tr, te = group_split(cell, 0)
ea = (fit_predict(x[tr], y[tr], x[te], 0, 30) - y[te]) ** 2       # 샘플별 제곱오차, (N_te,)
eb = (fit_predict(x[tr], y[tr], x[te], 0, 100) - y[te]) ** 2
d, n = eb - ea, len(ea)
cd = np.array([d[cell[te] == c].mean() for c in np.unique(cell[te])])   # 셀별 평균 차이, (C_te,)

print(f"MSE 차이 B - A = {d.mean():+.5f},  두 모델 오차의 상관 {np.corrcoef(ea, eb)[0, 1]:.2f}")
print(f"짝 없는 SE         {np.sqrt(ea.var(ddof=1) / n + eb.var(ddof=1) / n):.5f}")
print(f"짝지은 SE (샘플)   {d.std(ddof=1) / np.sqrt(n):.5f}")
print(f"짝지은 SE (셀 {len(cd)}개) {cd.std(ddof=1) / np.sqrt(len(cd)):.5f}")

# %% cell 4 — 반복 교차검증의 함정: 점수 수를 늘려도 불확실성은 줄지 않는다
def cv_scores(x, y, cell, reps, k=5):                     # 반복 group k-fold, 점수 reps·k개
    out = []
    for r in range(reps):
        fold = np.random.default_rng(r).permutation(np.unique(cell)) % k
        for j in range(k):
            te = fold[cell] == j
            out.append(np.sqrt(np.mean((fit_predict(x[~te], y[~te], x[te], r, 30) - y[te]) ** 2)))
    return np.array(out)

for reps in (1, 4, 16):
    est, naive = [], []
    for ds in range(100):                                 # 같은 분포에서 뽑은 독립 데이터셋 100개
        s = cv_scores(*make_data(np.random.default_rng(1000 + ds), n_cells=20, per_cell=10), reps)
        est.append(s.mean())
        naive.append(s.std(ddof=1) / np.sqrt(len(s)))
    print(f"반복 {reps:2d}번 (점수 {5 * reps:2d}개):  순진한 SE 평균 {np.mean(naive):.4f}   "
          f"CV 추정값의 실제 sd {np.std(est, ddof=1):.4f}")

# %% cell 5 — 여러 데이터셋에서의 비교: 평균 순위와 critical difference
from math import exp, sqrt

methods = {"A 특징 30": (30, 1.5), "B 특징 100": (100, 1.5), "C 특징 100, 좁은 대역": (100, 0.5)}
freqs = np.linspace(0.5, 3.0, 12)                          # 데이터셋 12개: 목표 함수의 진동수
scores = np.zeros((len(freqs), len(methods)))              # RMSE, (N_ds, k)
for i, fq in enumerate(freqs):
    xd, yd, cd_ = make_data(np.random.default_rng(i), freq=fq)
    tr, te = group_split(cd_, 0)
    for j, (nf, sc) in enumerate(methods.values()):
        scores[i, j] = np.sqrt(np.mean((fit_predict(xd[tr], yd[tr], xd[te], 0, nf, sc) - yd[te]) ** 2))

ranks = scores.argsort(axis=1).argsort(axis=1) + 1         # 데이터셋마다 순위 (1 = 최고), 동점 없음
N, k = scores.shape
avg = ranks.mean(axis=0)
chi2 = 12 * N / (k * (k + 1)) * (np.sum(avg**2) - k * (k + 1) ** 2 / 4)   # Friedman 통계량
cd_nemenyi = 2.343 * sqrt(k * (k + 1) / (6 * N))            # q_0.05 = 2.343 (k = 3)
for name, a, m in zip(methods, avg, scores.mean(axis=0)):
    print(f"{name:20s} 평균 순위 {a:.2f}   평균 RMSE {m:.4f}")
print(f"Friedman χ² = {chi2:.2f}, p = {exp(-chi2 / 2):.4f} (자유도 2)   Nemenyi CD = {cd_nemenyi:.2f}")
