"""17.6 실습의 공통 부분: 데이터, 모델 정의, 평가 함수. train.py와 eval.py가 함께 쓴다."""
import numpy as np
import torch
from torch import nn


def make_data(seed=0, n_train=512, n_val=256):
    g = np.random.default_rng(seed)
    temp_k = g.uniform(280.0, 360.0, size=n_train + n_val)
    pressure_bar = g.uniform(1.0, 5.0, size=n_train + n_val)
    y = np.sin(temp_k / 15.0) + 0.3 * pressure_bar ** 1.5 + g.normal(0, 0.05, size=temp_k.size)
    X = np.stack([temp_k, pressure_bar], axis=1).astype(np.float32)        # (N, 2)
    y = y.astype(np.float32)[:, None]                                     # (N, 1)
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


def build_model(cfg):
    return nn.Sequential(nn.Linear(2, cfg["hidden"]), nn.Tanh(), nn.Dropout(cfg["dropout"]),
                         nn.Linear(cfg["hidden"], cfg["hidden"]), nn.Tanh(), nn.Linear(cfg["hidden"], 1))


def evaluate(model, X, y, contract, batch_size=100):
    """평가 모드·no_grad에서 합과 개수를 누적해 MAE를 계산한다."""
    model.eval()
    mean, std = np.array(contract["x_mean"]), np.array(contract["x_std"])
    total, n = 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(X), batch_size):                           # 마지막 batch는 56개
            xb = torch.from_numpy((X[i:i + batch_size] - mean) / std).float()
            pred = model(xb) * contract["y_std"] + contract["y_mean"]
            total += (pred - torch.from_numpy(y[i:i + batch_size])).abs().sum().item()
            n += len(xb)
    return total / n
