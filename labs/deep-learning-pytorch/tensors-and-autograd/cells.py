"""17.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 4만 NumPy로 쓴 브라우저 셀이고(exec="browser"), 나머지는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/deep-learning-pytorch/tensors-and-autograd/cells.py
"""
# %% cell 1 — 텐서의 dtype·device와 NumPy 메모리 공유
import numpy as np
import torch

print(torch.tensor([1, 2]).dtype, torch.tensor([1.0, 2.0]).dtype,
      torch.from_numpy(np.zeros(2)).dtype)                 # 정수, Python 실수, NumPy float64
temp_c = np.array([20.0, 21.5, 19.0])                      # (3,) NumPy 배열
shared = torch.from_numpy(temp_c)                          # 메모리를 공유한다
copied = torch.tensor(temp_c)                              # 복사한다
shared[0] = -999.0
print("from_numpy 수정 후 원본:", temp_c)
copied[1] = -999.0
print("tensor() 수정 후 원본  :", temp_c)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.ones(2, 3, device=device)                        # (2, 3)
print(x.dtype, x.device.type, x.shape, x.to(torch.float16).dtype)

# %% cell 2 — view·reshape·transpose와 contiguous
x = torch.arange(6.0).reshape(2, 3)                        # (2, 3)
xt = x.t()                                                 # (3, 2) 전치: 같은 메모리, stride만 바뀜
print("stride:", x.stride(), xt.stride(), " contiguous:", x.is_contiguous(), xt.is_contiguous())
print("전치가 같은 메모리인가:", xt.data_ptr() == x.data_ptr())
try:
    xt.view(6)
except RuntimeError as e:
    print("xt.view(6) -> RuntimeError:", "reshape" in str(e))
r = xt.reshape(6)                                          # 필요하면 복사한다
print("xt.reshape(6):", r.tolist(), " 복사했는가:", r.data_ptr() != x.data_ptr())
v = x.view(3, 2)                                           # contiguous면 view가 된다
v[0, 0] = 100.0
print("view를 고치면 원본도 바뀐다:", x[0, 0].item())
img = torch.zeros(8, 32, 32, 3)                            # (B, H, W, C)
print("permute:", tuple(img.permute(0, 3, 1, 2).shape))    # (B, C, H, W)

# %% cell 3 — requires_grad, 계산 그래프, grad 누적, broadcasting의 gradient
w = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)     # (3,) 잎(leaf) 텐서
X = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]])      # (2, 3)
b = torch.zeros(3, requires_grad=True)                     # (3,) 두 행에 broadcast된다
loss = ((X * w + b) ** 2).sum()
print("grad_fn:", type(loss.grad_fn).__name__, " 잎인가:", w.is_leaf, (X * w).is_leaf)
loss.backward()
print("w.grad =", w.grad.tolist())
print("b.grad =", b.grad.tolist(), "= 2(Xw+b)를 batch 축으로 합한 값",
      (2 * (X * w + b)).sum(0).tolist())
loss2 = ((X * w + b) ** 2).sum()
loss2.backward()                                           # zero_grad 없이 한 번 더
print("두 번째 backward 후 w.grad =", w.grad.tolist(), "(두 배로 누적)")

# %% cell 4 — broadcasting의 gradient는 broadcast된 축의 합이다
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))                                # (B, D)
b = rng.normal(size=(1, 3))                                # (1, D) -> (B, D)로 broadcast
L = lambda b: np.sum(np.tanh(X + b) ** 2)                  # 스칼라 손실
dLdY = 2 * np.tanh(X + b) * (1 - np.tanh(X + b) ** 2)      # (B, D) 출력 쪽 gradient
grad_b = dLdY.sum(axis=0, keepdims=True)                   # (1, D) broadcast 축으로 합
num = np.zeros_like(b)
for j in range(3):                                         # 중앙차분으로 확인
    e = np.zeros_like(b); e[0, j] = 1e-6
    num[0, j] = (L(b + e) - L(b - e)) / 2e-6
print("합으로 줄인 gradient:", grad_b.round(6))
print("수치미분            :", num.round(6))
print("shape:", dLdY.shape, "->", grad_b.shape)

# %% cell 5 — detach, no_grad, inference_mode, in-place의 위험
w = torch.randn(3, requires_grad=True)
y = (w * 2).sum()
print("detach:", y.detach().requires_grad, " no_grad 안:", end=" ")
with torch.no_grad():
    print((w * 2).sum().requires_grad)
with torch.inference_mode():
    z = w * 2
try:
    (z * w).sum().backward()
except RuntimeError as e:
    print("inference_mode 텐서를 학습에 쓰면 RuntimeError:", "Inference tensors" in str(e))

a = torch.randn(3, requires_grad=True)
s = torch.sigmoid(a)                                       # backward에 s 자신이 필요하다
print("s 버전:", s._version, end=" -> ")
s.mul_(2)                                                  # in-place 수정
print(s._version)
try:
    s.sum().backward()
except RuntimeError as e:
    print("backward -> RuntimeError:", "modified by an inplace operation" in str(e))

# %% cell 6 — torch.func: grad·vmap·jacrev
from torch.func import grad, jacrev, vmap

torch.manual_seed(0)
w = torch.randn(3)                                         # (D,)
X, y = torch.randn(5, 3), torch.randn(5)                   # (N, D), (N,)
def loss_one(w, x, y):                                     # 표본 하나의 손실
    return (x @ w - y) ** 2

per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))(w, X, y)   # (N, D)
print("표본별 gradient shape:", tuple(per_sample.shape))
manual = 2 * (X @ w - y)[:, None] * X                      # (N, D) 손 유도
print("손 유도와 같은가:", torch.allclose(per_sample, manual))
f = lambda x: torch.tanh(x) * x.sum()                      # R^3 -> R^3
x0 = torch.tensor([0.1, -0.2, 0.3])
J = jacrev(f)(x0)                                          # (출력 3, 입력 3)
J_manual = torch.diag(1 - torch.tanh(x0) ** 2) * x0.sum() + torch.tanh(x0)[:, None]
print("Jacobian shape:", tuple(J.shape), " 손 유도와 같은가:", torch.allclose(J, J_manual))
