"""7.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linear-algebra/affine-and-nonlinear/cells.py
"""
# %% cell 1 — 원점의 상과 선형성 잔차
import numpy as np

A = np.array([[1.0, 0.5],
              [0.0, 1.0]])
b = np.array([0.5, 0.25])

def f(x):
    return A @ x + b

x = np.array([1.0, 2.0])
y = np.array([-1.0, 0.5])
print("f(0)             =", f(np.zeros(2)))
print("f(x+y)-f(x)-f(y) =", f(x + y) - f(x) - f(y))

# %% cell 2 — 동차좌표
M = np.block([[A, b[:, None]],
              [np.zeros((1, 2)), np.ones((1, 1))]])   # (3, 3)
x_h = np.append(x, 1.0)                                # (3,)
print(M @ x_h)
print(f(x))

# %% cell 3 — 두 층 접기
rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(4, 2)), rng.normal(size=4)   # (4, 2), (4,)
W2, b2 = rng.normal(size=(3, 4)), rng.normal(size=3)   # (3, 4), (3,)

W, c = W2 @ W1, W2 @ b1 + b2                           # (3, 2), (3,)
x = rng.normal(size=2)
print(np.allclose(W2 @ (W1 @ x + b1) + b2, W @ x + c))
print("rank(W) =", np.linalg.matrix_rank(W))

# %% cell 4 — ReLU 합의 구간별 기울기
def relu(z):
    return np.maximum(0.0, z)

def g(x):
    return relu(x) + relu(x - 1) + relu(x - 2)

h = 1e-6
for x0 in [-1.0, 0.5, 1.5, 2.5]:
    print(f"x = {x0:4.1f}  기울기 = {(g(x0 + h) - g(x0)) / h:.3f}")

# %% cell 5 — 국소 선형화의 오차
a = 0.5
def lin(x):
    return np.sin(a) + np.cos(a) * (x - a)    # x = a에서의 1차 근사

for d in [0.4, 0.2, 0.1, 0.05]:
    err = abs(np.sin(a + d) - lin(a + d))
    print(f"d = {d:4.2f}  오차 = {err:.2e}  오차/d^2 = {err / d**2:.4f}")
