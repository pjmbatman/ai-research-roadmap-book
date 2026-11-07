"""29.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/life-sciences/biology-for-ml/cells.py
"""
# %% cell 1 — 전사와 번역, 동의·비동의 변이
import numpy as np

BASES = "TCAG"
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"   # 표준 유전 부호
CODON = {a + b + c: AA[16 * i + 4 * j + k]
         for i, a in enumerate(BASES) for j, b in enumerate(BASES) for k, c in enumerate(BASES)}

def translate(dna):
    protein = ""
    for s in range(0, len(dna) - 2, 3):
        aa = CODON[dna[s:s + 3]]
        if aa == "*":                                  # 정지 코돈
            break
        protein += aa
    return protein

def point(seq, pos, base):                             # 한 자리 치환
    return seq[:pos] + base + seq[pos + 1:]

gene = "ATGGCTAAAGGTTGGTAA"                            # 코딩 가닥 5'→3'
print("mRNA        :", gene.replace("T", "U"))         # 전사: T 대신 U
print("단백질      :", translate(gene))
print("6번째 T→C   :", translate(point(gene, 5, "C")))   # GCT→GCC
print("11번째 G→A  :", translate(point(gene, 10, "A")))  # GGT→GAT
n_stop = sum(v == "*" for v in CODON.values())
print(f"코돈 {len(CODON)}개 = 아미노산 지정 {len(CODON) - n_stop} + 정지 {n_stop}")

# %% cell 2 — 시퀀싱 깊이와 기술·생물학적 반복
rng = np.random.default_rng(0)
G = 2000                                               # 유전자 수
frac = rng.lognormal(0, 1.5, G)
frac /= frac.sum()                                     # (2000,) 시료의 실제 mRNA 비율
bio = frac * rng.lognormal(0, 0.3, G)
bio /= bio.sum()                                       # (2000,) 같은 조건의 다른 개체

def logcorr(a, b):
    return np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]

for depth in [10_000, 1_000_000]:                      # 시료 하나에서 읽은 read 수
    rep1 = rng.poisson(depth * frac)                   # (2000,) 기술 반복 1
    rep2 = rng.poisson(depth * frac)                   # (2000,) 기술 반복 2
    other = rng.poisson(depth * bio)                   # (2000,) 생물학적 반복
    print(f"depth {depth:>9,}: 0인 유전자 {np.mean(rep1 == 0):4.0%}  "
          f"기술 반복 r {logcorr(rep1, rep2):.3f}  생물학적 반복 r {logcorr(rep1, other):.3f}")

# %% cell 3 — 배치와 조건이 겹친 실험 설계
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestCentroid
warnings.filterwarnings("ignore", module="threadpoolctl")      # 브라우저 실행 환경의 무해한 경고만 끈다

def make_study(confounded, n=40, genes=200):
    cond = np.repeat([0, 1], n // 2)                   # (40,) 대조군 0, 처리군 1
    batch = cond.copy() if confounded else np.tile([0, 1], n // 2)
    shift = rng.normal(0, 0.5, genes)                  # (200,) 배치 1의 유전자별 측정 편향
    X = rng.normal(0, 1, (n, genes)) + np.outer(batch, shift)   # (40, 200) 조건 효과는 0
    return X, cond

for confounded in [True, False]:
    X, cond = make_study(confounded)
    acc = cross_val_score(NearestCentroid(), X, cond, cv=5).mean()
    print(f"배치와 조건 {'일치' if confounded else '교차'}: 조건 분류 정확도 {acc:.2f}")
