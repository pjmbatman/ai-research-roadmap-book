"""8.4 실습: 직접 만든 자동미분 엔진(engine.py)으로 작은 MLP를 학습하고 PyTorch autograd와 비교한다.

1. 같은 가중치에서 엔진·PyTorch·유한차분 gradient를 비교한다.
2. 엔진과 PyTorch로 같은 SGD 학습을 돌려 손실 곡선과 최종 파라미터를 비교한다.

PyTorch가 필요하다.
실행: uv run --project labs --group torch python labs/calculus-and-autodiff/automatic-differentiation/micrograd_vs_torch.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_moons

sys.path.insert(0, str(Path(__file__).parent))
from engine import MLP, Value  # noqa: E402

SIZES = [2, 8, 8, 1]
X, y01 = make_moons(n_samples=40, noise=0.1, random_state=0)
Y = 2.0 * y01 - 1.0                                   # 레이블 {-1, +1}


def engine_loss(model):
    """평균 logistic loss log(1 + exp(-y f(x)))."""
    total = Value(0.0)
    for x, y in zip(X, Y):
        f = model(x)[0]
        total = total + ((f * (-y)).exp() + 1.0).log()
    return total / len(X)


def torch_model_from(model):
    """엔진 MLP와 같은 가중치를 가진 PyTorch 파라미터 리스트(float64)."""
    params = []
    for W, b in model.layers:
        params.append(torch.tensor([[w.data for w in row] for row in W], dtype=torch.float64, requires_grad=True))
        params.append(torch.tensor([bi.data for bi in b], dtype=torch.float64, requires_grad=True))
    return params


def torch_loss(params):
    h = torch.tensor(X, dtype=torch.float64)          # (N, 2)
    n_layers = len(params) // 2
    for i in range(n_layers):
        W, b = params[2 * i], params[2 * i + 1]
        h = h @ W.T + b                               # (N, out)
        if i < n_layers - 1:
            h = torch.tanh(h)
    f = h[:, 0]                                       # (N,)
    return torch.nn.functional.softplus(-torch.tensor(Y) * f).mean()


def engine_grads(model):
    out = []
    for W, b in model.layers:
        out.append(np.array([[w.grad for w in row] for row in W]))
        out.append(np.array([bi.grad for bi in b]))
    return out


def compare_gradients():
    print("[1] 같은 가중치에서 gradient 비교 (MLP 2-8-8-1, 표본 40개)")
    model = MLP(SIZES, seed=0)
    loss = engine_loss(model)
    model.zero_grad()
    loss.backward()
    g_engine = engine_grads(model)

    params = torch_model_from(model)
    lt = torch_loss(params)
    lt.backward()
    print(f"  손실: 엔진 {loss.data:.12f}   PyTorch {lt.item():.12f}")
    names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    for name, ge, p in zip(names, g_engine, params):
        diff = np.abs(ge - p.grad.numpy()).max()
        print(f"  {name} {str(tuple(ge.shape)):8}  최대 |엔진 - PyTorch| = {diff:.1e}")

    # 유한차분: 첫 층 가중치 몇 개
    eps = 1e-6
    for (i, j) in [(0, 0), (3, 1), (7, 0)]:
        w = model.layers[0][0][i][j]
        old = w.data
        w.data = old + eps
        lp = engine_loss(model).data
        w.data = old - eps
        lm = engine_loss(model).data
        w.data = old
        fd = (lp - lm) / (2 * eps)
        print(f"  W1[{i},{j}]  엔진 {g_engine[0][i, j]:+.8f}  유한차분 {fd:+.8f}")


def train_both(steps=200, lr=1.0):
    print(f"[2] SGD {steps} step (학습률 {lr}): 엔진 vs PyTorch")
    model = MLP(SIZES, seed=0)
    params = torch_model_from(model)
    for step in range(steps + 1):
        loss = engine_loss(model)
        model.zero_grad()
        loss.backward()
        lt = torch_loss(params)
        for p in params:
            p.grad = None
        lt.backward()
        if step % 50 == 0:
            f = np.array([model(x)[0].data for x in X])
            acc = np.mean(np.sign(f) == Y)
            pdiff = max(np.abs(a - p.detach().numpy()).max() for a, p in zip(engine_params(model), params))
            print(f"  step {step:3d}  손실 엔진 {loss.data:.6f}  PyTorch {lt.item():.6f}"
                  f"  정확도 {acc:.3f}  파라미터 차이 {pdiff:.1e}")
        if step == steps:
            break
        for p in model.parameters():
            p.data -= lr * p.grad
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
    # 학습이 끝난 엔진의 가중치를 PyTorch에 복사해 같은 점에서 gradient를 다시 비교한다
    same = torch_model_from(model)
    torch_loss(same).backward()
    gdiff = max(np.abs(a - p.grad.numpy()).max() for a, p in zip(engine_grads(model), same))
    print(f"  학습 후 엔진의 가중치에서 gradient 최대 차이 {gdiff:.1e}")


def engine_params(model):
    out = []
    for W, b in model.layers:
        out.append(np.array([[w.data for w in row] for row in W]))
        out.append(np.array([bi.data for bi in b]))
    return out


if __name__ == "__main__":
    compare_gradients()
    train_both()
