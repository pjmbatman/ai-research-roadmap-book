"""29.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/life-sciences/sequence-models/cells.py
"""
# %% cell 1 — one-hot과 k-mer 표현
from itertools import product

import numpy as np

DNA = "ACGT"

def one_hot(seq, alphabet=DNA):
    idx = np.array([alphabet.index(c) for c in seq])
    return np.eye(len(alphabet))[idx]                  # (L, |A|)

def kmer_counts(seq, k, alphabet=DNA):
    vocab = {"".join(p): i for i, p in enumerate(product(alphabet, repeat=k))}
    v = np.zeros(len(vocab))                           # (|A|^k,)
    for s in range(len(seq) - k + 1):
        v[vocab[seq[s:s + k]]] += 1
    return v

a, b = "ACGTACGTTT", "TTACGTACGT"                      # 같은 조각을 다른 순서로 배치
print("one-hot shape:", one_hot(a).shape)
for k in [1, 2, 3]:
    same = np.array_equal(kmer_counts(a, k), kmer_counts(b, k))
    print(f"k={k}: 차원 {4**k:3d}, 두 서열의 k-mer 벡터가 같다: {same}")
print("단백질(20종) 3-mer 차원:", 20**3)

# %% cell 2 — 합성 단백질 가족과 마스크 예측 손실
AA = "ACDEFGHIKLMNPQRSTVWY"
HYD = np.array([c in "AILMFVW" for c in AA])           # (20,) 소수성 아미노산
rng = np.random.default_rng(0)

def family(label, n=8, L=80, rate=0.3):                # 조상 서열 + 무작위 치환으로 만든 가족
    p = np.where(HYD, 1.0 + 1.0 * label, 1.0)
    anc = rng.choice(20, L, p=p / p.sum())             # (80,) label 1이면 소수성 잔기가 많다
    kids = np.tile(anc, (n, 1))                        # (8, 80)
    hit = rng.random(kids.shape) < rate
    kids[hit] = rng.integers(0, 20, hit.sum())
    return ["".join(AA[i] for i in s) for s in kids]

labels = rng.integers(0, 2, 40)                        # (40,) 가족 40개의 기능 label
seqs = [s for f in labels for s in family(f)]
fam = np.repeat(np.arange(40), 8)                      # (320,) 가족 번호
S = np.array([[AA.index(c) for c in s] for s in seqs])   # (320, 80)
uni = np.bincount(S[fam < 30].ravel(), minlength=20) + 1.0   # 가족 0–29 = 학습 데이터
probs = {"균등": np.full((80, 20), 0.05), "전체 빈도": np.tile(uni / uni.sum(), (80, 1))}
loss = {k: [] for k in [*probs, "가족 프로필"]}
for f in range(30, 40):                                # 학습에 없던 가족의 서열
    query, rest = S[fam == f][0], S[fam == f][1:]      # (80,), (7, 80)
    prof = np.stack([np.bincount(c, minlength=20) for c in rest.T]) + 0.5
    probs["가족 프로필"] = prof / prof.sum(1, keepdims=True)   # (80, 20) 위치별 분포
    mask = rng.random(80) < 0.15                       # 15%를 가린다
    for k, P in probs.items():                         # 가린 자리의 평균 음의 로그우도
        loss[k].append(-np.mean(np.log(P[mask, query[mask]])))
for k, v in loss.items():
    print(f"{k}: 가린 자리 손실 {np.mean(v):.2f} nats")

# %% cell 3 — 공진화 신호: 열 쌍의 상호정보량
N, C = 1000, 30
msa = rng.integers(0, 20, (N, C))                      # (1000, 30) 독립인 열들
partner = rng.permutation(20)                          # 5번 열의 잔기 → 20번 열의 짝 잔기
keep = rng.random(N) < 0.9
msa[keep, 20] = partner[msa[keep, 5]]                  # 90%의 서열에서 짝을 맞춘다

def mutual_info(x, z):
    joint = np.zeros((20, 20))
    np.add.at(joint, (x, z), 1)
    joint /= joint.sum()
    px, pz = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return np.sum(joint[nz] * np.log(joint[nz] / (px @ pz)[nz]))

mi = {(i, j): mutual_info(msa[:, i], msa[:, j]) for i in range(C) for j in range(i + 1, C)}
top = sorted(mi, key=mi.get, reverse=True)[:3]
for p in top:
    print(f"열 {p}: MI {mi[p]:.2f} nats")
print(f"전체 {len(mi)}쌍의 MI 중앙값 {np.median(list(mi.values())):.2f} nats")

# %% cell 4 — 상동성 누수: 무작위 분할과 가족 분할
import warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore", module="threadpoolctl")      # 브라우저 실행 환경의 무해한 경고만 끈다

y = labels[fam]                                        # (320,)
X1 = np.stack([kmer_counts(s, 1, AA) for s in seqs])   # (320, 20) 조성
X2 = np.stack([kmer_counts(s, 2, AA) for s in seqs])   # (320, 400)

def nn1(Xtr, ytr, Xte):                                # 가장 가까운 train 서열의 label을 베낀다
    d = (Xte**2).sum(1)[:, None] + (Xtr**2).sum(1)[None, :] - 2 * Xte @ Xtr.T   # (n_te, n_tr)
    return ytr[d.argmin(1)]

def logreg(Xtr, ytr, Xte):
    return LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, ytr).predict(Xte)

def cv_acc(fit_predict, X, folds):                     # folds: (320,) 샘플마다 fold 번호 0–4
    return np.mean([np.mean(fit_predict(X[folds != k], y[folds != k], X[folds == k]) == y[folds == k])
                    for k in range(5)])

random_folds = np.random.default_rng(1).permutation(320) % 5   # 서열 단위로 무작위
family_folds = fam % 5                                         # 가족 단위로 통째로
for name, fp, X in [("1-NN, 2-mer", nn1, X2), ("로지스틱, 조성", logreg, X1)]:
    print(f"{name}: 무작위 분할 {cv_acc(fp, X, random_folds):.2f}   "
          f"가족 분할 {cv_acc(fp, X, family_folds):.2f}")
