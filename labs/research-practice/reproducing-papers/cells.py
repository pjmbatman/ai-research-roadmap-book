"""23.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-practice/reproducing-papers/cells.py
"""
# %% cell 1 — 가상의 논문과 공개 코드
import numpy as np

rng = np.random.default_rng(0)
G, M = 40, 25                                       # 셀 40개, 셀마다 측정 25개
group = np.repeat(np.arange(G), M)                  # 각 측정의 셀 번호, (N,)
desc = rng.normal(size=(G, 12))[group]              # 셀 단위 설명 변수, 셀 안에서 일정, (N, 12)
meas = rng.normal(size=(G * M, 6)) * [1, 1, 1, 1, 100, 100]   # 측정 특징, 두 열은 단위가 큼, (N, 6)
y = meas @ [0.5, -0.3, 0.2, 0.1, 0.004, -0.003] + rng.normal(0, 0.6, G)[group] + rng.normal(0, 0.3, G * M)
X_base, X_prop = meas, np.c_[meas, desc]           # baseline: 측정만, 제안: 측정 + 셀 설명, (N, 6), (N, 18)

def ridge_rmse(X, seed, group_split, standardize, lam_mean):
    r = np.random.default_rng(seed)
    if group_split:                                 # 시험 셀 10개를 통째로 뗀다
        te = np.isin(group, r.permutation(G)[:10])
    else:                                           # 측정 단위로 25%를 무작위로 뗀다
        te = r.permutation(G * M) < G * M // 4
    Xtr, Xte, ytr, yte = X[~te], X[te], y[~te], y[te]
    if standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    A, B = np.c_[Xtr, np.ones(len(Xtr))], np.c_[Xte, np.ones(len(Xte))]
    lam = len(Xtr) if lam_mean else 1.0             # 평균 손실의 lam=1은 합 손실의 lam=N
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    return np.sqrt(np.mean((B @ w - yte) ** 2))

print("측정", len(y), "개, 셀", G, "개, 특징: baseline", X_base.shape[1], "/ 제안", X_prop.shape[1])

# %% cell 2 — 공식 코드 조건에서 seed 10개로 재현하고 보고값과 비교
from scipy import stats

code = dict(group_split=False, standardize=False, lam_mean=False)   # 공개 코드의 실제 설정
runs = {"baseline": np.array([ridge_rmse(X_base, s, **code) for s in range(10)]),
        "제안": np.array([ridge_rmse(X_prop, s, **code) for s in range(10)])}
runs["차이"] = runs["제안"] - runs["baseline"]       # 같은 seed끼리 짝지은 차이, (10,)
reported = {"baseline": 0.69, "제안": 0.51, "차이": -0.18}   # 논문 표 1 (가상, 단일 실행)

t = stats.t.ppf(0.975, df=9)
for name, r in runs.items():
    half = t * r.std(ddof=1) * np.sqrt(1 + 1 / len(r))   # 새 단일 실행의 95% 예측구간 반폭
    lo, hi = r.mean() - half, r.mean() + half
    inside = "안" if lo <= reported[name] <= hi else "밖"
    print(f"{name:<8} 재현 {r.mean():.3f} ± {r.std(ddof=1):.3f}  "
          f"단일 실행 구간 [{lo:.3f}, {hi:.3f}]  보고 {reported[name]:.3f} → 구간 {inside}")

# %% cell 3 — 논문 본문과 공개 코드의 불일치 세 가지를 2^3 조합으로 분리
import itertools

names = ["group_split", "standardize", "lam_mean"]
cfgs = list(itertools.product([0, 1], repeat=3))   # 8개 조합, (8, 3)
D = []
print("셀분할 표준화 평균손실 | baseline  제안   차이(제안-baseline)")
for cfg in cfgs:
    kw = dict(zip(names, map(bool, cfg)))
    b = np.array([ridge_rmse(X_base, s, **kw) for s in range(10)])
    p = np.array([ridge_rmse(X_prop, s, **kw) for s in range(10)])
    D.append((p - b).mean())
    o = ["-O"[c] for c in cfg]
    print(f"  {o[0]}      {o[1]}      {o[2]}     | {b.mean():.3f}    {p.mean():.3f}  "
          f"{(p - b).mean():+.3f} ± {(p - b).std(ddof=1):.3f}")
S, D = 2 * np.array(cfgs) - 1, np.array(D)         # -1/+1 부호 (8, 3), 조합별 평균 차이 (8,)
for j, k in enumerate(names):
    print(f"{k:<12} 주효과 {D @ S[:, j] / 4:+.3f}")
print(f"group_split × lam_mean 교호작용 {D @ (S[:, 0] * S[:, 2]) / 4:+.3f}")
