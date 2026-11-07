"""8.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 4는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/calculus-and-autodiff/automatic-differentiation/cells.py
"""
# %% cell 1 — 유한차분의 오차: 간격 h와 부동소수점 정밀도
import numpy as np

def f(x):
    return np.exp(np.sin(x))

def f32(x):
    """float32 계산: 함수값을 float32로 반올림해 저장한다(float32 exp·sin의 마지막 비트는 구현마다 다르다)"""
    return np.float32(f(np.float64(x)))

x0 = 1.0
exact = np.cos(x0) * np.exp(np.sin(x0))               # 해석적 도함수
print("     h      float64 오차   float32 오차")
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]:
    d64 = (f(x0 + h) - f(x0 - h)) / (2 * h)            # 중앙차분
    x32, h32 = np.float32(x0), np.float32(h)
    d32 = (f32(x32 + h32) - f32(x32 - h32)) / (2 * h32)
    print(f"{h:8.0e}   {abs(d64 - exact):12.2e}   {abs(float(d32) - exact):12.2e}")

# %% cell 2 — 30줄짜리 reverse mode 자동미분 엔진
import math

class Value:
    """스칼라 값 하나와 그 gradient를 담는 계산 그래프의 노드"""
    def __init__(self, data, parents=(), local_grads=()):
        self.data, self.grad = data, 0.0
        self.parents, self.local_grads = parents, local_grads   # 입력 노드와 국소 미분

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t**2,))

    def backward(self):
        order = []
        def visit(v):                      # 위상 정렬: 입력이 출력보다 먼저 온다
            if v not in order:
                for p in v.parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):          # 출력에서 입력 쪽으로
            for p, local in zip(v.parents, v.local_grads):
                p.grad += local * v.grad   # chain rule, 여러 경로의 기여는 더한다

# %% cell 3 — 엔진, 손 유도, 유한차분 비교
x1, x2 = Value(2.0), Value(-1.0)
w1, w2, b = Value(-0.5), Value(1.5), Value(0.8)
y = (x1 * w1 + x2 * w2 + b).tanh()                  # 뉴런 하나
y.backward()

s = 1 - y.data**2                                   # tanh'(a)
def y_of(w1_):                                      # 유한차분용: w1만 바꾼 함수
    return math.tanh(2.0 * w1_ + (-1.0) * 1.5 + 0.8)
fd = (y_of(-0.5 + 1e-6) - y_of(-0.5 - 1e-6)) / 2e-6
print(f"y = {y.data:.6f}")
print(f"dy/dw1  엔진 {w1.grad:+.6f}  손 유도 {s * x1.data:+.6f}  유한차분 {fd:+.6f}")
print(f"dy/dw2  엔진 {w2.grad:+.6f}  손 유도 {s * x2.data:+.6f}")
print(f"dy/db   엔진 {b.grad:+.6f}  손 유도 {s:+.6f}")

u = Value(3.0)
z = u * u + u                                       # 같은 노드를 세 번 쓴다
z.backward()
print(f"z = u*u + u, dz/du 엔진 {u.grad}  손 유도 2u + 1 = {2 * 3.0 + 1}")

# %% cell 4 — PyTorch autograd와 비교, Hessian-vector product
import torch

x = torch.tensor([2.0, -1.0], dtype=torch.float64)
w = torch.tensor([-0.5, 1.5], dtype=torch.float64, requires_grad=True)
b = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
y = torch.tanh(w @ x + b)                           # 같은 뉴런
y.backward()
print(f"y = {y.item():.6f}")
print("dy/dw =", w.grad.numpy().round(6), "  dy/db =", round(b.grad.item(), 6))

w.grad = None
L = torch.tanh(w @ x + 0.8) ** 2                    # 스칼라 손실
(g,) = torch.autograd.grad(L, w, create_graph=True) # gradient도 그래프로 남긴다
v = torch.tensor([1.0, 0.0], dtype=torch.float64)
(Hv,) = torch.autograd.grad(g @ v, w)               # d(g·v)/dw = H v
t = torch.tanh(w @ x + 0.8).item()
H_exact = 2 * (1 - t**2) * (1 - 3 * t**2) * torch.outer(x, x)   # 손 유도 Hessian (2, 2)
print("H v (autograd) =", Hv.numpy().round(6))
print("H v (손 유도)  =", (H_exact @ v).numpy().round(6))

# %% cell 5 — 깊은 합성에서 gradient의 크기
rng = np.random.default_rng(0)
D, L = 64, 30
for gain in [0.5, 1.0, 3.0]:
    h = rng.normal(size=D)                          # (D,) 입력
    Js = []
    for _ in range(L):
        W = rng.normal(size=(D, D)) * gain / np.sqrt(D)   # (D, D)
        h = np.tanh(W @ h)
        Js.append((1 - h**2)[:, None] * W)          # 층의 Jacobian (D, D)
    g = np.ones(D) / np.sqrt(D)                     # 출력 쪽 gradient, 노름 1
    norms = []
    for J in reversed(Js):                          # 역전파: VJP를 차례로
        g = J.T @ g
        norms.append(np.linalg.norm(g))
    print(f"gain {gain}: 층 30→1의 |grad|", " ".join(f"{n:.1e}" for n in norms[4::5]))
