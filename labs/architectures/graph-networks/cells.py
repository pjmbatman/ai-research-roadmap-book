"""18.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/architectures/graph-networks/cells.py
"""
# %% cell 1 — 간선 목록에서 인접행렬과 이웃 합
import numpy as np

edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (2, 6), (6, 7)]  # 6원자 고리 + 곁사슬
N = 8
A = np.zeros((N, N))                                    # 인접행렬 (N, N)
for i, j in edges:
    A[i, j] = A[j, i] = 1.0                             # 무방향 그래프: 대칭
types = np.array([0, 0, 0, 0, 0, 1, 0, 2])              # 0 = C, 1 = N, 2 = O
X = np.eye(3)[types]                                    # 노드 특징: 원소 one-hot (N, 3)

print("차수     :", A.sum(1).astype(int))
print("대칭     :", np.array_equal(A, A.T), "  간선 수:", int(A.sum() / 2))
print("이웃 합 A @ X  (열 = 이웃의 C, N, O 개수)")
print((A @ X).astype(int))

A_hat = A + np.eye(N)                                   # self-loop 추가
d = A_hat.sum(1)
A_norm = A_hat / np.sqrt(np.outer(d, d))                # D^-1/2 (A+I) D^-1/2 (N, N)
print("A_norm[2, 6] =", round(A_norm[2, 6], 4), " = 1/sqrt(4*3):", round(1 / np.sqrt(12), 4))

# %% cell 2 — message passing 층의 순열 등변성과 집계 함수
rng = np.random.default_rng(0)
W_self, W_nei = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))   # 노드 수와 무관한 가중치

def aggregate(A, H, agg="sum"):                         # 이웃 특징 모으기 (N, D_in)
    M = A @ H                                           # 이웃의 합
    return M / np.maximum(A.sum(1, keepdims=True), 1) if agg == "mean" else M

def mp_layer(A, H, agg="sum"):                          # (N, N), (N, D_in) → (N, D_out)
    return np.maximum(0, H @ W_self + aggregate(A, H, agg) @ W_nei)

P = np.eye(N)[rng.permutation(N)]                       # 노드 번호 바꾸기 (N, N)
H1 = mp_layer(A, X)
print("등변  f(PAP^T, PX) = P f(A, X):", np.allclose(mp_layer(P @ A @ P.T, P @ X), P @ H1))
print("불변  readout 합              :", np.allclose(mp_layer(P @ A @ P.T, P @ X).sum(0), H1.sum(0)))
W_flat = rng.normal(size=(N * 3, 4))                    # 펼친 특징에 쓰는 MLP 층
print("펼친 MLP 불변                 :", np.allclose((P @ X).ravel() @ W_flat, X.ravel() @ W_flat))

# 중심 노드 0의 이웃이 탄소 1개인 그래프 vs 탄소 2개인 그래프
A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0.]]); A2 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0.]])
Xc = np.tile(np.eye(3)[[0]], (3, 1))                    # 모든 노드가 탄소 (3, 3)
for agg in ["sum", "mean"]:
    same = np.allclose(aggregate(A1, Xc, agg)[0], aggregate(A2, Xc, agg)[0])
    print(f"{agg:4} 집계: 이웃 1개와 2개를 {'구별 못 함' if same else '구별함'}")

# %% cell 3 — 그래프 구성 비교: 빠진 센서 값을 이웃 평균으로 채우기
rng = np.random.default_rng(2)
S, T_train, T_test = 30, 200, 100                       # 센서 30개, 학습·평가 시점 수
pos_m = rng.random((S, 2)) * 100.0                      # 센서 위치 (S, 2), 단위 m
k_rad_m = rng.normal(size=(6, 2)) * 0.04                # 공간 파수 6개 (6, 2), 단위 rad/m
phase = rng.random(6) * 2 * np.pi
def field(T, k_rad_m):                                  # 모드 진폭만 시점마다 바뀌는 매끄러운 장 (T, S)
    amp = rng.normal(size=(T, 6))
    return amp @ np.sin(pos_m @ k_rad_m.T + phase).T
Y_train, Y_test = field(T_train, k_rad_m), field(T_test, k_rad_m)
Y_shift = field(T_test, k_rad_m[:, ::-1] * [1, -1])     # 파수 벡터를 90° 돌린 새 조건

def knn_graph(Dist, k=4):                               # 각 노드에서 가장 가까운 k개로 연결 (S, S)
    G = np.zeros((S, S))
    G[np.arange(S)[:, None], np.argsort(Dist, 1)[:, 1:k + 1]] = 1.0
    return G

dist_m = np.linalg.norm(pos_m[:, None] - pos_m[None], axis=-1)     # 물리적 거리 (S, S)
corr = np.corrcoef(Y_train.T)                                      # 학습 구간의 센서 간 상관 (S, S)
graphs = {"물리적 거리 kNN": knn_graph(dist_m), "학습 상관 kNN  ": knn_graph(-corr),
          "완전 연결      ": 1.0 - np.eye(S), "무작위 4-이웃  ": knn_graph(rng.random((S, S)) - np.eye(S))}
mae = lambda Y, G: np.abs(Y @ G.T / G.sum(1) - Y).mean()          # 각 센서를 이웃 평균으로 예측
print("그래프            같은 조건 MAE   돌린 조건 MAE")
for name, G in graphs.items():
    print(f"{name}       {mae(Y_test, G):.3f}          {mae(Y_shift, G):.3f}")
print(f"(참고) 0 예측         {np.abs(Y_test).mean():.3f}          {np.abs(Y_shift).mean():.3f}")

# %% cell 4 — over-smoothing: 전파를 반복하면 노드 특징이 같아진다
rng = np.random.default_rng(1)
n, D = 40, 8                                            # 공동체 2개 × 20 노드, 특징 8차원
y = np.repeat([0, 1], n // 2)
p_edge = np.where(y[:, None] == y[None, :], 0.25, 0.05) # 같은 공동체끼리 더 자주 연결
A = np.triu(rng.random((n, n)) < p_edge, 1).astype(float)
A = A + A.T
mu = rng.normal(size=(2, D))
X = mu[y] + 2.0 * rng.normal(size=(n, D))               # 잡음이 큰 노드 특징 (n, D)

A_hat = A + np.eye(n); d = A_hat.sum(1)
A_norm = A_hat / np.sqrt(np.outer(d, d))                # (n, n)

def centroid_acc(H):                                    # 공동체 평균에 가까운 쪽으로 분류
    c = np.stack([H[y == k].mean(0) for k in (0, 1)])
    return (np.argmin(((H[:, None] - c) ** 2).sum(-1), 1) == y).mean()

H = X.copy()
for k in range(65):
    if k in (0, 1, 2, 4, 8, 16, 32, 64):
        Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
        cos = (Hn @ Hn.T)[y[:, None] != y[None, :]].mean()   # 다른 공동체 노드 사이 cosine
        print(f"전파 {k:2d}번  분류 정확도 {centroid_acc(H):.2f}   다른 공동체 간 cos {cos:+.3f}")
    H = A_norm @ H                                      # 학습 파라미터 없는 GCN 전파
