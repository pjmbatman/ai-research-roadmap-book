"""17.7 실습의 과제 정의: 데이터, 모델, step 함수, 평가 함수. trainer.py를 import하지 않는다."""
import torch
from torch import nn

CONFIG = {"hidden": 32, "lr": 3e-3, "batch_size": 32, "seed": 0}


def make_data(seed=0, n=640):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g) * 4 - 2                         # (N, 2)
    y = torch.sin(2 * x[:, :1]) * torch.cos(x[:, 1:]) + 0.05 * torch.randn(n, 1, generator=g)   # (N, 1)
    return (x[:512], y[:512]), (x[512:], y[512:])


def build_model(cfg):
    return nn.Sequential(nn.Linear(2, cfg["hidden"]), nn.Tanh(), nn.Linear(cfg["hidden"], cfg["hidden"]),
                         nn.Tanh(), nn.Linear(cfg["hidden"], 1))


def batches(x, y, batch_size):
    return list(zip(x.split(batch_size), y.split(batch_size)))


def regression_step(model, batch):
    x, y = batch
    return nn.functional.mse_loss(model(x), y)


def make_eval_fn(x, y, batch_size=50):
    """평가 모드 전환, gradient 끄기, 합/개수 누적을 모두 여기서 책임진다(17.6장)."""
    def eval_fn(model):
        was_training = model.training
        model.eval()
        se, ae, n = 0.0, 0.0, 0
        with torch.inference_mode():
            for xb, yb in batches(x, y, batch_size):
                r = model(xb) - yb
                se, ae, n = se + r.pow(2).sum().item(), ae + r.abs().sum().item(), n + len(xb)
        model.train(was_training)
        return {"val_rmse": (se / n) ** 0.5, "val_mae": ae / n}
    return eval_fn
