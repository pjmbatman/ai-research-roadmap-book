"""17.9 실습: 버그 다섯 개를 켜고 끌 수 있는 작은 분류 학습 파이프라인.

bugs는 아래 이름의 집합이다. 실제 연구 코드에서 흔한 실수를 하나씩 흉내 낸다.
    label-misalign   입력을 섞을 때 label은 섞지 않는다
    no-normalize     입력을 표준화하지 않는다 (온도 K처럼 큰 값이 그대로 들어간다)
    double-softmax   모델 끝에 softmax를 붙이고 CrossEntropyLoss(내부에서 log-softmax)를 쓴다
    no-zero-grad     step마다 gradient를 지우지 않는다
    eval-mode        평가할 때 model.eval()을 부르지 않는다 (dropout이 켜진 채)
"""
import math

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)                    # 작은 모델: 스레드 하나가 더 빠르고, 공유 서버의 CPU를 덜 쓴다

BUGS = ("label-misalign", "no-normalize", "double-softmax", "no-zero-grad", "eval-mode")
N_CLASSES = 3


def load_data(bugs, seed=0, n=900):
    """합성 물성 분류: (온도 K, 압력 bar, 조성 비율) → 상(phase) 3종."""
    g = np.random.default_rng(seed)
    centers = np.array([[300.0, 2.0, 0.2], [330.0, 3.0, 0.5], [360.0, 2.5, 0.8]])
    scales = np.array([12.0, 0.5, 0.12])
    y = g.integers(0, N_CLASSES, size=n)
    X = centers[y] + scales * g.normal(size=(n, 3))                        # (N, 3)
    perm = g.permutation(n)
    X = X[perm]
    if "label-misalign" not in bugs:
        y = y[perm]                                                        # 버그: 이 줄이 빠진다
    Xtr, ytr, Xva, yva = X[:600], y[:600], X[600:], y[600:]
    if "no-normalize" not in bugs:
        mu, sd = Xtr.mean(0), Xtr.std(0)                                   # 학습 데이터 통계만 쓴다
        Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    t = lambda a, dt: torch.tensor(a, dtype=dt)
    return t(Xtr, torch.float32), t(ytr, torch.long), t(Xva, torch.float32), t(yva, torch.long)


def build_model(bugs, seed=0):
    torch.manual_seed(seed)
    layers = [nn.Linear(3, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, N_CLASSES)]
    if "double-softmax" in bugs:
        layers.append(nn.Softmax(dim=1))                                   # 버그: 확률을 다시 log-softmax
    return nn.Sequential(*layers)


def train_step(model, opt, xb, yb, bugs):
    model.train()
    loss = nn.functional.cross_entropy(model(xb), yb)
    if "no-zero-grad" not in bugs:
        opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def evaluate(model, X, y, bugs):
    if "eval-mode" not in bugs:
        model.eval()
    with torch.no_grad():
        logits = model(X)
        acc = (logits.argmax(1) == y).float().mean().item()
        loss = nn.functional.cross_entropy(logits, y).item()
    return loss, acc


def train(bugs, epochs=30, lr=1e-2, batch_size=32, seed=0):
    Xtr, ytr, Xva, yva = load_data(bugs, seed)
    model = build_model(bugs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xtr), generator=g).split(batch_size):
            train_step(model, opt, Xtr[idx], ytr[idx], bugs)
    return evaluate(model, Xva, yva, bugs)


def chance_loss():
    return math.log(N_CLASSES)
