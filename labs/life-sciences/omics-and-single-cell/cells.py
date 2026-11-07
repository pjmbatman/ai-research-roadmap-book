"""29.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/life-sciences/omics-and-single-cell/cells.py
"""
# %% cell 1 — 합성 단일세포 개수 행렬과 희소성
import numpy as np

rng = np.random.default_rng(0)
G, THETA = 500, 2.0                                    # 유전자 수, 음이항 분포의 모양 매개변수

def simulate(n_per_type, response=1.0):
    types = np.repeat(np.arange(3), n_per_type)        # (N,) 세포 유형 0, 1, 2
    mu = np.tile(base, (3, 1))                         # (3, 500) 유형별 평균 발현
    for t in range(3):
        mu[t, 20 * t:20 * t + 20] *= 8                 # 유형마다 marker 유전자 20개
    mu[2, 100:120] *= response                         # 유형 2만 반응하는 유전자(예: 처리 조건)
    size = rng.lognormal(0.0, 0.6, len(types))         # (N,) 세포마다 포획 효율
    lam = size[:, None] * mu[types]                    # (N, 500)
    return rng.poisson(rng.gamma(THETA, lam / THETA)), types   # 감마–Poisson = 음이항

base = rng.lognormal(0.0, 1.5, G)                     # (500,) 유전자별 기본 발현 수준
counts, types = simulate([200, 200, 200])              # (600, 500)
lib = counts.sum(1)                                    # (600,) 세포별 총 개수
lam_hat = lib[:, None] * counts.sum(0)[None, :] / counts.sum()   # (600, 500) 단순 기대값
print(f"0의 비율 {np.mean(counts == 0):.2f}")
print(f"세포별 총 개수: 최소 {lib.min()}, 중앙값 {int(np.median(lib))}, 최대 {lib.max()}")
print(f"Poisson이 예측하는 0의 비율 {np.mean(np.exp(-lam_hat)):.2f}")

# %% cell 2 — 정규화, log1p, 고변이 유전자, PCA, 군집
import warnings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari
warnings.filterwarnings("ignore", module="threadpoolctl")      # 브라우저 실행 환경의 무해한 경고만 끈다

def normalize(C):                                      # 세포마다 총합 1만으로 맞춘 뒤 log1p
    return np.log1p(C / C.sum(1, keepdims=True) * 1e4)          # (N, G)

def hvg(X, n=100):                                     # 평균 대비 분산이 큰 유전자 n개의 번호
    m, v = X.mean(0), X.var(0)
    return np.argsort(v / (m + 1e-3))[-n:]                       # (n,)

def embed_and_cluster(M, k=3):
    Z = PCA(10, random_state=0).fit_transform(M - M.mean(0))     # (N, 10)
    return Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z)

norm = normalize(counts)                               # (600, 500)
for name, M in [("원 개수", counts.astype(float)), ("정규화+log1p+HVG", norm[:, hvg(norm)])]:
    Z, lab = embed_and_cluster(M)
    r = np.corrcoef(np.abs(Z[:, 0]), lib)[0, 1]
    print(f"{name}: |PC1|과 총 개수의 상관 {r:.2f}, 군집과 세포 유형의 ARI {ari(types, lab):.2f}")

# %% cell 3 — 배치 효과와 과보정
counts2, types2 = simulate([40, 200, 160], response=4.0)   # 배치 2: 구성이 다르고 유형 2가 반응
T = np.concatenate([types, types2])                    # (1000,) 세포 유형
B = np.repeat([0, 1], [600, 400])                      # (1000,) 배치
truth = normalize(np.vstack([counts, counts2]))        # (1000, 500) 배치 효과가 없을 때
M = truth.copy()
M[B == 1] += rng.normal(0, 0.8, G)                     # 로그 공간에서 유전자마다 더해지는 배치 효과
genes = hvg(M)                                         # (100,) 군집에 쓸 유전자는 한 번만 고른다

def center(M, ref):                                    # 배치마다 ref 세포의 평균을 뺀다
    out = M.copy()
    for b in (0, 1):
        out[B == b] -= M[(B == b) & ref].mean(0)
    return out

def response(X, t):                                    # 유형 t에서 반응 유전자 점수의 배치 간 차이
    r = X[:, 100:120].mean(1)
    return r[(T == t) & (B == 1)].mean() - r[(T == t) & (B == 0)].mean()

for name, X in [("참값(배치 효과 없음)", truth), ("보정 전", M),
                ("배치 평균 빼기", center(M, np.ones(len(B), bool))),
                ("공유 유형 1로 보정", center(M, T == 1))]:
    _, lab = embed_and_cluster(X[:, genes])
    print(f"{name}: ARI(유형) {ari(T, lab):.2f} ARI(배치) {ari(B, lab):.2f} | "
          f"반응 차이 유형0 {response(X, 0):+.2f} 유형2 {response(X, 2):+.2f}")

# %% cell 4 — 계수 분포: 평균–분산 관계
X0 = counts[types == 0]                                # (200, 500) 한 유형의 세포
m, v = X0.mean(0), X0.var(0)
keep = m > 0.5
inv_theta = np.sum((v - m)[keep] * m[keep] ** 2) / np.sum(m[keep] ** 4)   # v ≈ m + m²/θ 최소제곱
print(f"분산/평균 중앙값 {np.median(v[keep] / m[keep]):.1f}  (Poisson이면 1)")
print(f"추정 θ {1 / inv_theta:.2f}  (시뮬레이션의 θ = {THETA})")
