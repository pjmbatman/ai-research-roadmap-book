"""16.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/classical-ml/unsupervised-learning/cells.py
"""
# %% cell 1 — PCA는 섞인 신호를 풀지 못하고 ICA는 푼다
import warnings
import numpy as np
from sklearn.decomposition import PCA, FastICA

warnings.filterwarnings("ignore", module="threadpoolctl")      # 브라우저 실행 환경의 무해한 경고만 끈다

t_s = np.linspace(0, 8, 2000)                                   # (T,) 시간 [s]
S = np.c_[np.sin(2 * np.pi * 1.0 * t_s),                       # 원천 1: 사인파
          np.sign(np.sin(2 * np.pi * 0.3 * t_s))]              # 원천 2: 사각파, (T, 2)
A = np.array([[1.0, 0.6], [0.5, 1.0]])                         # 혼합 행렬 (센서 2개)
X = S @ A.T                                                    # (T, 2) 관측

def match(Z):                                                  # 각 원천과 가장 닮은 성분의 |상관|
    C = np.abs(np.corrcoef(S.T, Z.T)[:2, 2:])                  # (원천, 성분)
    return C.max(axis=1).round(2)

Z_pca = PCA(n_components=2).fit_transform(X)                   # (T, 2)
Z_ica = FastICA(n_components=2, random_state=0).fit_transform(X)
print("관측 센서    |상관|:", match(X))
print("PCA 성분     |상관|:", match(Z_pca))
print("ICA 성분     |상관|:", match(Z_ica))

# %% cell 2 — 스케일이 군집을 바꾼다
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(3)
label = np.repeat([0, 1, 2], 100)                              # 참 집단 (N,)
temp_k = np.array([300.0, 310.0, 320.0])[label] + rng.normal(0, 2, 300)    # 온도 [K]
press_pa = 1e5 + rng.normal(0, 3000, 300)                      # 압력 [Pa]: 집단과 무관한 변동
F = np.c_[temp_k, press_pa]                                    # (N, 2)

for name, Fx in [("K, Pa", F), ("K, kPa", F / [1, 1000]), ("표준화", StandardScaler().fit_transform(F))]:
    pred = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Fx)   # (N,)
    print(f"{name:7s} ARI = {adjusted_rand_score(label, pred):.2f}")

# %% cell 3 — 군집 수 고르기: 구조가 있을 때와 없을 때
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

data = {"덩어리 3개": np.r_[rng.normal([0, 0], 0.5, (100, 2)),
                            rng.normal([3, 0], 0.5, (100, 2)),
                            rng.normal([0, 3], 0.5, (100, 2))],
        "균일 분포": rng.uniform(0, 4, (300, 2))}              # 각각 (300, 2)
for name, D in data.items():
    sil = [silhouette_score(D, KMeans(k, n_init=10, random_state=0).fit_predict(D)) for k in range(2, 7)]
    bic = [GaussianMixture(k, random_state=0).fit(D).bic(D) for k in range(1, 7)]
    print(f"{name}: silhouette k=2..6 {np.round(sil, 2)}  최대 k={2 + int(np.argmax(sil))}"
          f" | GMM BIC 최소 k={1 + int(np.argmin(bic))}")

# %% cell 4 — 축마다 정상인 이상치
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

cov = np.array([[1.0, 0.95], [0.95, 1.0]])
normal = rng.multivariate_normal([0, 0], cov, 500)             # (500, 2) 강하게 상관된 정상 데이터
u = rng.uniform(0.7, 1.5, 10) * rng.choice([-1, 1], 10)
odd = np.c_[u, -u]                                             # (10, 2) 상관을 거스르는 점들
Xa = np.r_[normal, odd]                                        # (510, 2)
is_odd = np.r_[np.zeros(500), np.ones(10)]

mu, sd = normal.mean(0), normal.std(0)
score_z = np.abs((Xa - mu) / sd).max(axis=1)                   # 축별 z-점수의 최댓값
pc = PCA(1).fit(normal)                                        # 정상 데이터로만 적합
score_pca = np.sum((Xa - pc.inverse_transform(pc.transform(Xa))) ** 2, axis=1)
score_if = -IsolationForest(random_state=0).fit(normal).score_samples(Xa)
for name, s in [("축별 z-점수", score_z), ("PCA 재구성 오차", score_pca), ("isolation forest", score_if)]:
    print(f"{name:16s} AUC = {roc_auc_score(is_odd, s):.3f}")

# %% cell 5 — t-SNE의 군집 크기와 거리
from sklearn.manifold import TSNE

blobs = np.r_[rng.normal(0, 0.2, (100, 5)),                    # 촘촘한 군집 A
              rng.normal(0, 2.0, (100, 5)) + [6, 0, 0, 0, 0],  # 10배 퍼진 군집 B
              rng.normal(0, 0.2, (100, 5)) + [40, 0, 0, 0, 0]] # 멀리 떨어진 군집 C, (300, 5)
E = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(blobs)

def spread(Z, i):
    return np.linalg.norm(Z[i] - Z[i].mean(0), axis=1).mean()

def gap(Z, i, j):
    return np.linalg.norm(Z[i].mean(0) - Z[j].mean(0))

a, b, c = np.arange(0, 100), np.arange(100, 200), np.arange(200, 300)
for name, Z in [("원공간", blobs), ("t-SNE", E)]:
    print(f"{name:5s}  퍼짐 B/A = {spread(Z, b) / spread(Z, a):3.0f}배   거리 AC/AB = {gap(Z, a, c) / gap(Z, a, b):3.0f}배")
