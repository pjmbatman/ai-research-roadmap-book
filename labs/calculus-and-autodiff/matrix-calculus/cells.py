"""8.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/calculus-and-autodiff/matrix-calculus/cells.py
"""
# %% cell 1 — 항등식을 수치미분으로 확인
import numpy as np

def num_grad(f, X, eps=1e-6):
    """스칼라 함수 f의 X에 대한 gradient를 중앙차분으로. 결과 shape = X.shape"""
    G = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        E = np.zeros_like(X)
        E[idx] = eps
        G[idx] = (f(X + E) - f(X - E)) / (2 * eps)
    return G

rng = np.random.default_rng(0)
A = rng.normal(size=(3, 3))                       # (3, 3) 대칭이 아님
x = rng.normal(size=3)                            # (3,)
X = rng.normal(size=(3, 3)) + 3 * np.eye(3)       # (3, 3) det > 0

checks = {
    "x^T A x  -> (A + A^T) x": (num_grad(lambda v: v @ A @ v, x), (A + A.T) @ x),
    "tr(A X)  -> A^T        ": (num_grad(lambda M: np.trace(A @ M), X), A.T),
    "log det X -> X^{-T}    ": (num_grad(lambda M: np.log(np.linalg.det(M)), X), np.linalg.inv(X).T),
}
for name, (num, formula) in checks.items():
    print(f"{name}  최대 차이 {np.abs(num - formula).max():.1e}")

# %% cell 2 — 선형회귀의 gradient
N, D = 50, 3
Xd = rng.normal(size=(N, D))                      # (N, D) 설계행렬
y = Xd @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=N)   # (N,)
w = np.zeros(D)                                   # (D,)

def mse(w):
    r = Xd @ w - y                                # (N,) 잔차
    return 0.5 * np.mean(r**2)

grad_w = Xd.T @ (Xd @ w - y) / N                  # (D, N)(N,) = (D,)
print("손 유도  :", grad_w.round(6))
print("수치미분 :", num_grad(mse, w).round(6))

# %% cell 3 — softmax cross-entropy의 gradient = p - y
def softmax(z):
    e = np.exp(z - z.max())                       # 큰 logit에서의 overflow 방지
    return e / e.sum()

z = np.array([2.0, 1.0, -1.0, 0.5])               # (K,) logits
y_onehot = np.array([0.0, 1.0, 0.0, 0.0])         # (K,) 정답은 인덱스 1

ce = lambda z: -np.sum(y_onehot * np.log(softmax(z)))
p = softmax(z)
print("p - y    :", (p - y_onehot).round(6))
print("수치미분 :", num_grad(ce, z).round(6))
print("성분 합  :", f"{abs((p - y_onehot).sum()):.1e}")

# %% cell 4 — 2층 MLP의 역전파를 손으로
N, D, H, K = 8, 3, 5, 4
Xb = rng.normal(size=(N, D))                      # (N, D)
labels = rng.integers(0, K, size=N)               # (N,)
Y = np.eye(K)[labels]                             # (N, K) one-hot
P0 = {"W1": rng.normal(size=(H, D)), "b1": rng.normal(size=H),
      "W2": rng.normal(size=(K, H)), "b2": rng.normal(size=K)}

def forward(P):
    A1 = Xb @ P["W1"].T + P["b1"]                 # (N, H)
    Hh = np.maximum(A1, 0)                        # (N, H)
    Z = Hh @ P["W2"].T + P["b2"]                  # (N, K)
    Z = Z - Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)   # (N, K)
    return A1, Hh, Pr, -np.mean(np.sum(Y * np.log(Pr), axis=1))

A1, Hh, Pr, loss = forward(P0)
Gz = (Pr - Y) / N                                 # (N, K) dL/dZ
Ga = (Gz @ P0["W2"]) * (A1 > 0)                   # (N, H) dL/dA1
grads = {"W2": Gz.T @ Hh, "b2": Gz.sum(0), "W1": Ga.T @ Xb, "b1": Ga.sum(0)}
for k, g in grads.items():
    f = lambda v, k=k: forward({**P0, k: v})[3]
    g_num = num_grad(f, P0[k])
    rel = np.linalg.norm(g - g_num) / (np.linalg.norm(g) + np.linalg.norm(g_num))
    print(f"{k}: shape {str(g.shape):7}  상대오차 {rel:.1e}")
