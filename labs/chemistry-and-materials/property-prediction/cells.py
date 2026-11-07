"""28.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/chemistry-and-materials/property-prediction/cells.py
"""
# %% cell 1 — 측정 잡음이 정하는 오차 바닥
import numpy as np

rng = np.random.default_rng(0)
y_true = rng.normal(-3.0, 1.5, size=20000)                 # 참 물성(예: log 용해도) (N,)
for sigma in [0.2, 0.5, 0.8]:
    y_lab = y_true + sigma * rng.normal(size=y_true.size)    # 실험 레이블 = 참값 + 측정 잡음
    mae_perfect = np.abs(y_true - y_lab).mean()              # 참값을 정확히 아는 모델의 MAE
    r2_ceiling = 1 - ((y_lab - y_true) ** 2).mean() / y_lab.var()
    print(f"잡음 σ = {sigma:.1f}  완벽한 모델의 MAE = {mae_perfect:.3f}"
          f" (σ√(2/π) = {sigma * np.sqrt(2 / np.pi):.3f})  R² 상한 = {r2_ceiling:.3f}")

# %% cell 2 — 골격 구조가 있는 합성 화학 공간
n_bits, n_fam, n_scaf, n_mol = 256, 12, 150, 1200
fam_bits = [rng.choice(n_bits, 10, replace=False) for _ in range(n_fam)]   # 골격 계열마다 공유 비트
fam_of = rng.choice(n_fam, n_scaf, p=np.arange(n_fam, 0, -1) / np.arange(n_fam, 0, -1).sum())
scaf_bits = [np.union1d(fam_bits[fam_of[k]], rng.choice(n_bits, 6, replace=False)) for k in range(n_scaf)]
sub_bits = [rng.choice(n_bits, 4, replace=False) for _ in range(50)]       # 치환기 50종
w = rng.normal(0, 0.15, n_bits)                                            # 비트마다 기여
fam_eff = rng.normal(0, 1.0, n_fam)                                        # 계열 효과
scaf_eff = rng.normal(0, 0.3, n_scaf)                                      # 골격 고유 효과
p_scaf = 1 / np.arange(1, n_scaf + 1) ** 0.8                               # 골격 크기가 치우친 분포
scaf = rng.choice(n_scaf, n_mol, p=p_scaf / p_scaf.sum())                  # (N,)
X = np.zeros((n_mol, n_bits), dtype=bool)                                  # 지문 (N, n_bits)
for i, s in enumerate(scaf):
    X[i, scaf_bits[s]] = True
    for t in rng.choice(50, rng.integers(1, 5), replace=False):
        X[i, sub_bits[t]] = True
y = X @ w + fam_eff[fam_of[scaf]] + scaf_eff[scaf] + 0.2 * rng.normal(size=n_mol)   # 물성 (N,)

def tanimoto(A, B):                                                        # (n, d), (m, d) → (n, m)
    inter = A.astype(float) @ B.T.astype(float)
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter)

S = tanimoto(X, X)
same = scaf[:, None] == scaf[None, :]
off = ~np.eye(n_mol, dtype=bool)
print(f"분자 {n_mol}개, 골격 {len(np.unique(scaf))}개, 가장 큰 골격 3개의 크기 {np.sort(np.bincount(scaf))[::-1][:3]}")
print(f"평균 Tanimoto: 같은 골격 {S[same & off].mean():.3f}, 다른 골격 {S[~same].mean():.3f}")

# %% cell 3 — 무작위 분할과 scaffold 분할
def fit_predict(tr, te, lam=0.1):                          # Tanimoto 커널 GP의 사후 평균·표준편차
    K = tanimoto(X[tr], X[tr]) + lam * np.eye(len(tr))     # (n_tr, n_tr)
    Ks = tanimoto(X[te], X[tr])                            # (n_te, n_tr)
    mu = y[tr].mean()
    mean = mu + Ks @ np.linalg.solve(K, y[tr] - mu)
    var = 1.0 - np.einsum("ij,ji->i", Ks, np.linalg.solve(K, Ks.T))
    return mean, np.sqrt(np.clip(var, 1e-12, None)) * y[tr].std()

perm = rng.permutation(n_mol)
splits = {"무작위": (perm[240:], perm[:240])}
test_scafs, n_te = [], 0
for s in rng.permutation(n_scaf):                          # 골격 단위로 test를 채운다
    if n_te < 240:
        test_scafs.append(s)
        n_te += (scaf == s).sum()
is_te = np.isin(scaf, test_scafs)
splits["scaffold"] = (np.flatnonzero(~is_te), np.flatnonzero(is_te))
for name, (tr, te) in splits.items():
    mean, sd = fit_predict(tr, te)
    nn_sim = tanimoto(X[te], X[tr]).max(1)                 # test 분자의 최근접 train 유사도
    print(f"{name:8s} train {len(tr)} test {len(te)}  MAE {np.abs(mean - y[te]).mean():.3f}"
          f"  평균 예측 MAE {np.abs(y[tr].mean() - y[te]).mean():.3f}  최근접 유사도 중앙값 {np.median(nn_sim):.3f}")

# %% cell 4 — 적용 영역: 유사도와 불확실성으로 본 오차
from scipy.stats import spearmanr

tr, te = splits["scaffold"]
mean, sd = fit_predict(tr, te)
err = np.abs(mean - y[te])
nn_sim = tanimoto(X[te], X[tr]).max(1)
bins = np.digitize(nn_sim, [0.4, 0.5])                    # 0: < 0.4, 1: 0.4–0.5, 2: ≥ 0.5
for b, label in enumerate(["< 0.4  ", "0.4–0.5", "≥ 0.5  "]):
    print(f"최근접 유사도 {label}  분자 {(bins == b).sum():3d}  MAE {err[bins == b].mean():.3f}")
print(f"Spearman(예측 표준편차, |오차|) = {spearmanr(sd, err)[0]:.3f}")
keep = sd <= np.quantile(sd, 0.5)
print(f"불확실성 하위 50%만 남기면 MAE {err[keep].mean():.3f} (전체 {err.mean():.3f})")
