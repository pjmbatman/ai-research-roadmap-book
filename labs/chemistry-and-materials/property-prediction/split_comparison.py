"""28.2 실습: 합성 화학 공간에서 무작위·scaffold·계열(군집) 분할의 성능 차이와 불확실성–오차 관계를 여러 시드로 잰다.

데이터 생성 규칙은 본문 cell 2와 같다(골격 계열 12개, 골격 150개, 분자 1200개, 256비트 지문).
모델은 Tanimoto 커널 GP(사후 평균·표준편차)다.

실행: uv run --project labs python labs/chemistry-and-materials/property-prediction/split_comparison.py
"""
import numpy as np
from scipy.stats import spearmanr


def make_data(seed, n_bits=256, n_fam=12, n_scaf=150, n_mol=1200):
    rng = np.random.default_rng(seed)
    fam_bits = [rng.choice(n_bits, 10, replace=False) for _ in range(n_fam)]
    fam_of = rng.choice(n_fam, n_scaf, p=np.arange(n_fam, 0, -1) / np.arange(n_fam, 0, -1).sum())
    scaf_bits = [np.union1d(fam_bits[fam_of[k]], rng.choice(n_bits, 6, replace=False)) for k in range(n_scaf)]
    sub_bits = [rng.choice(n_bits, 4, replace=False) for _ in range(50)]
    w = rng.normal(0, 0.15, n_bits)
    fam_eff = rng.normal(0, 1.0, n_fam)
    scaf_eff = rng.normal(0, 0.3, n_scaf)
    p_scaf = 1 / np.arange(1, n_scaf + 1) ** 0.8
    scaf = rng.choice(n_scaf, n_mol, p=p_scaf / p_scaf.sum())
    X = np.zeros((n_mol, n_bits), dtype=bool)
    for i, s in enumerate(scaf):
        X[i, scaf_bits[s]] = True
        for t in rng.choice(50, rng.integers(1, 5), replace=False):
            X[i, sub_bits[t]] = True
    y = X @ w + fam_eff[fam_of[scaf]] + scaf_eff[scaf] + 0.2 * rng.normal(size=n_mol)
    return X, y, scaf, fam_of[scaf], rng


def tanimoto(A, B):
    inter = A.astype(float) @ B.T.astype(float)
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter)


def gp(X, y, tr, te, lam=0.1):
    K = tanimoto(X[tr], X[tr]) + lam * np.eye(len(tr))
    Ks = tanimoto(X[te], X[tr])
    mu = y[tr].mean()
    mean = mu + Ks @ np.linalg.solve(K, y[tr] - mu)
    var = 1.0 - np.einsum("ij,ji->i", Ks, np.linalg.solve(K, Ks.T))
    return mean, np.sqrt(np.clip(var, 1e-12, None)) * y[tr].std()


def group_split(groups, rng, frac=0.2):
    test, n = [], 0
    for g in rng.permutation(np.unique(groups)):
        if n < frac * len(groups):
            test.append(g)
            n += (groups == g).sum()
    is_te = np.isin(groups, test)
    return np.flatnonzero(~is_te), np.flatnonzero(is_te)


def main():
    rows = {k: [] for k in ["무작위", "scaffold", "계열"]}
    for seed in range(5):
        X, y, scaf, fam, rng = make_data(seed)
        perm = rng.permutation(len(y))
        splits = {"무작위": (perm[240:], perm[:240]), "scaffold": group_split(scaf, rng), "계열": group_split(fam, rng)}
        for name, (tr, te) in splits.items():
            mean, sd = gp(X, y, tr, te)
            err = np.abs(mean - y[te])
            rho = spearmanr(sd, err)[0]
            sim = np.median(tanimoto(X[te], X[tr]).max(1))
            rows[name].append((err.mean(), np.abs(y[tr].mean() - y[te]).mean(), sim, rho, len(te)))
    print("시드 0–4 평균 ± 표준편차")
    print(f"{'분할':8s} {'test 수':>8s} {'MAE':>14s} {'평균 예측 MAE':>14s} {'최근접 유사도':>12s} {'Spearman(sd,|e|)':>18s}")
    for name, r in rows.items():
        r = np.array(r)
        m, s = r.mean(0), r.std(0)
        print(f"{name:8s} {m[4]:8.0f} {m[0]:7.3f} ± {s[0]:.3f} {m[1]:7.3f} ± {s[1]:.3f} {m[2]:12.3f} {m[3]:10.3f} ± {s[3]:.3f}")


if __name__ == "__main__":
    main()
