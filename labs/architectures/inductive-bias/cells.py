"""18.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/architectures/inductive-bias/cells.py
"""
# %% cell 1 — 합성곱은 제약을 건 행렬이다
import numpy as np

T, k = 8, 3
def conv_matrix(kernel, T):                          # 원형 합성곱을 (T, T) 행렬로
    M = np.zeros((T, T))
    for t in range(T):
        for j, w in enumerate(kernel):
            M[t, (t + j - 1) % T] = w                # 출력 t는 입력 t-1, t, t+1만 본다
    return M

rng = np.random.default_rng(0)
M_conv = conv_matrix(np.array([1.0, -2.0, 1.0]), T)  # (T, T), 이산 2계 차분
M_dense = rng.normal(size=(T, T))                    # 아무 가정 없는 dense 층 (T, T)
print(M_conv[:3].astype(int))

S = np.roll(np.eye(T), 1, axis=0)                    # 한 칸 이동 행렬 (T, T)
x = rng.normal(size=T)                               # (T,)
relu = lambda z: np.maximum(0.0, z)
for name, M in [("conv ", M_conv), ("dense", M_dense)]:
    equi = np.allclose(M @ (S @ x), S @ (M @ x))
    inv = np.isclose(relu(M @ (S @ x)).sum(), relu(M @ x).sum())
    print(f"{name}  등변 f(Sx) = S f(x): {equi!s:5}  ReLU 후 합 불변: {inv}")
print("자유 파라미터  dense =", T * T, " 국소(공유 없음) =", T * k, " 합성곱 =", k)

# %% cell 2 — 편향이 맞을 때: 데이터 크기에 따른 성능
T, K = 32, 3
def target(X, w_pos):                                # (N, T) → (N,), 이웃 두 점의 국소 상호작용 합
    return np.maximum(0, X * np.roll(X, -1, axis=1)) @ w_pos / np.sqrt(T)

A_mlp, b_mlp = rng.normal(size=(T, 300)) / np.sqrt(T), rng.normal(size=300)
A_cnn, b_cnn = rng.normal(size=(K, 100)), rng.normal(size=100)
models = [
    lambda X: X,                                                  # 선형 (N, T)
    lambda X: relu(X @ A_mlp + b_mlp),                            # MLP 특징 (N, 300), 위치 구조 없음
    lambda X: relu(np.stack([np.roll(X, -j, axis=1) for j in range(K)], -1)
                    @ A_cnn + b_cnn).sum(axis=1),                 # 합성곱 특징 (N, 100), 국소·공유·합 pooling
]
def r2(feat, Xtr, ytr, Xte, yte, lam=1.0):           # ridge 회귀로 마지막 층만 학습
    F, G = np.c_[feat(Xtr), np.ones(len(Xtr))], np.c_[feat(Xte), np.ones(len(Xte))]
    w = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ ytr)
    return 1 - np.mean((G @ w - yte) ** 2) / np.var(yte)

def curves(w_pos, sizes=(20, 100, 1000, 3000)):
    Xte = rng.normal(size=(1000, T))
    print("    N   test R²: 선형     MLP    합성곱")
    for N in sizes:
        Xtr = rng.normal(size=(N, T))
        row = [r2(f, Xtr, target(Xtr, w_pos), Xte, target(Xte, w_pos)) for f in models]
        print(f"{N:5d}          " + "  ".join(f"{v:6.3f}" for v in row))

curves(np.ones(T))                                   # 위치와 무관한 목표

# %% cell 3 — 편향이 틀릴 때: 위치에 의존하는 목표
curves(np.linspace(0, 2, T))                         # 뒤쪽 위치일수록 가중치가 큰 목표
