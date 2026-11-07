"""28.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/chemistry-and-materials/chemistry-evaluation/cells.py
"""
# %% cell 1 — 정확한 중복과 근접 중복 탐지
import numpy as np

rng = np.random.default_rng(0)
L = 64
n_core = 120
cores = (rng.random((n_core, L)) < 0.12).astype(bool)      # 서로 다른 핵심 구조 120개

def make_dataset(n):
    X, origin = [], []
    for _ in range(n):
        c = rng.integers(n_core)
        x = cores[c].copy()
        for t in rng.choice(L, rng.integers(0, 3), replace=False):   # 치환기 0–2개 추가
            x[t] = True
        X.append(x); origin.append(c)
    return np.array(X), np.array(origin)

X, origin = make_dataset(1500)
keys = {x.tobytes(): i for i, x in enumerate(X)}
n_exact = len(X) - len(keys)                                 # 비트가 완전히 같은 중복
print(f"분자 {len(X)}개, 서로 다른 비트 벡터 {len(keys)}개 → 정확한 중복 {n_exact}개")

def tanimoto(A, B):
    inter = A.astype(float) @ B.T.astype(float)
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)

S = tanimoto(X, X); np.fill_diagonal(S, 0)
near = (S >= 0.9).any(1)                                     # Tanimoto ≥ 0.9인 짝이 있는 분자
print(f"근접 중복(Tanimoto ≥ 0.9인 짝이 있는 분자): {near.sum()}개 ({near.mean():.0%})")
print(f"같은 핵심 구조에서 나온 분자 쌍의 평균 유사도: {S[origin[:, None] == origin[None]].mean():.3f}")

# %% cell 2 — 분할 간 누수: 근접 중복이 train과 test에 갈라지면
w = rng.normal(0, 1.0, L)
y = X.astype(float) @ w + 0.1 * rng.normal(size=len(X))      # 물성 (N,)

def knn_predict(tr, te):                                     # 가장 가까운 train 분자(들)의 평균(암기 baseline)
    Sij = tanimoto(X[te], X[tr])
    mx = Sij.max(1)                                           # 최근접 유사도
    co_max = Sij >= mx[:, None] - 1e-9                        # 동점인 최근접 이웃 모두(순서 무관)
    return (co_max * y[tr][None]).sum(1) / co_max.sum(1), mx

perm = rng.permutation(len(X))
tr, te = perm[500:], perm[:500]
pred, nn_sim = knn_predict(tr, te)
leaked = nn_sim >= 0.9                                       # test 분자가 train에 근접 중복을 둔 경우
mae_all = np.abs(pred - y[te]).mean()
mae_clean = np.abs(pred - y[te])[~leaked].mean()
print(f"무작위 분할, 1-NN 암기 baseline (물성 표준편차 {y.std():.2f})")
print(f"  전체 test MAE            {mae_all:.3f}")
print(f"  근접 중복 test 비율       {leaked.mean():.0%}")
print(f"  근접 중복 제거 후 MAE     {mae_clean:.3f}  (누수가 성능을 {mae_clean / mae_all:.1f}배 좋아 보이게 했다)")

# %% cell 3 — 벤치마크 함정: 암기 baseline이 '최신 모델'을 흉내 낸다
from numpy.linalg import lstsq

def ridge_fit(tr, te, lam=1.0):
    A = np.c_[X[tr].astype(float), np.ones(len(tr))]
    beta = np.linalg.solve(A.T @ A + lam * np.eye(L + 1), A.T @ y[tr])
    return np.c_[X[te].astype(float), np.ones(len(te))] @ beta

for name, keep in [("원본(중복 포함)", np.ones(len(te), bool)), ("근접 중복 제거 후", ~leaked)]:
    idx = te[keep]
    nn_pred, _ = knn_predict(tr, idx)
    ridge_pred = ridge_fit(tr, idx)
    print(f"{name:16s}  1-NN 암기 MAE {np.abs(nn_pred - y[idx]).mean():.3f}   Ridge 모델 MAE {np.abs(ridge_pred - y[idx]).mean():.3f}")

# %% cell 4 — 지표의 오차 막대: 두 모델 차이가 유의한가
def bootstrap_mae_diff(err_a, err_b, n_boot=2000):
    rg = np.random.default_rng(1)
    diffs = []
    idx = np.arange(len(err_a))
    for _ in range(n_boot):
        b = rg.choice(idx, len(idx), replace=True)
        diffs.append(err_a[b].mean() - err_b[b].mean())
    return np.percentile(diffs, [2.5, 97.5])

clean = te[~leaked]
err_nn = np.abs(knn_predict(tr, clean)[0] - y[clean])
err_ridge = np.abs(ridge_fit(tr, clean) - y[clean])
lo, hi = bootstrap_mae_diff(err_nn, err_ridge)
print(f"근접 중복 제거 test에서 MAE 차이(1-NN − Ridge): {err_nn.mean() - err_ridge.mean():+.3f}")
print(f"  95% 부트스트랩 구간 [{lo:+.3f}, {hi:+.3f}]  → 0을 포함하지 않음: {not (lo <= 0 <= hi)}")

# %% cell 5 — 계산 수준을 섞으면 모델이 물리 대신 출처를 배운다
level = rng.integers(0, 2, len(X))                           # 0, 1 두 계산 수준
offset = np.array([0.0, 3.0])                                # 수준마다 체계적 차이
y_mixed = X.astype(float) @ w + offset[level] + 0.1 * rng.normal(size=len(X))
tr2, te2 = perm[500:], perm[:500]
# (a) 수준을 무시하고 섞어 학습·평가
beta = np.linalg.solve(np.c_[X[tr2].astype(float), np.ones(len(tr2))].T @ np.c_[X[tr2].astype(float), np.ones(len(tr2))]
                       + np.eye(L + 1), np.c_[X[tr2].astype(float), np.ones(len(tr2))].T @ y_mixed[tr2])
pred_mixed = np.c_[X[te2].astype(float), np.ones(len(te2))] @ beta
mae_mixed = np.abs(pred_mixed - y_mixed[te2]).mean()
# (b) 한 수준만 써서 학습·평가
m0 = level == 0
tr0 = np.array([i for i in tr2 if m0[i]]); te0 = np.array([i for i in te2 if m0[i]])
A0 = np.c_[X[tr0].astype(float), np.ones(len(tr0))]
beta0 = np.linalg.solve(A0.T @ A0 + np.eye(L + 1), A0.T @ y_mixed[tr0])
mae_single = np.abs(np.c_[X[te0].astype(float), np.ones(len(te0))] @ beta0 - y_mixed[te0]).mean()
print(f"두 계산 수준(체계적 차이 {offset[1] - offset[0]:.1f})을 섞어 학습·평가한 MAE: {mae_mixed:.3f}")
print(f"한 수준만 써서 학습·평가한 MAE:                 {mae_single:.3f}")
print(f"섞었을 때 늘어난 오차의 상당 부분은 물리가 아니라 계산 수준 차이에서 온다")
