"""17.6 실습: 중단·재개가 끊김 없는 학습과 같은 결과를 내는지 확인하는 작은 학습 스크립트.

    uv run --project labs --group torch python labs/deep-learning-pytorch/evaluation-and-checkpoints/train.py \
        --out labs/deep-learning-pytorch/evaluation-and-checkpoints/outputs/straight

옵션
    --steps N       총 학습 step 수(기본 300)
    --stop-at K     K step에서 last.pt를 저장하고 멈춘다(중단 흉내)
    --resume PATH   checkpoint에서 이어서 학습한다

데이터는 시드로 만든 합성 회귀 문제다. 입력 두 개(온도 K, 압력 bar)의 스케일이 크게 다르므로
학습 데이터의 평균·표준편차로 표준화하고, 그 통계를 checkpoint의 inference contract에 저장한다.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import build_model, evaluate, make_data

CONFIG = {"hidden": 32, "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-4, "batch_size": 32, "seed": 0}


def save(path, model, opt, sched, gen, step, best, contract):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "rng": {"torch": torch.get_rng_state(), "batch": gen.get_state(),
                        "numpy": np.random.get_state(), "python": random.getstate()},
                "step": step, "best": best, "config": CONFIG, "contract": contract}, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--stop-at", type=int)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    (X, y), (Xv, yv) = make_data()
    contract = {"inputs": ["temp_k", "pressure_bar"], "x_mean": X.mean(0).tolist(), "x_std": X.std(0).tolist(),
                "y_mean": float(y.mean()), "y_std": float(y.std())}
    Xn = torch.from_numpy((X - X.mean(0)) / X.std(0)).float()              # (N, 2)
    yn = torch.from_numpy((y - y.mean()) / y.std()).float()                # (N, 1)

    torch.manual_seed(CONFIG["seed"])
    model = build_model(CONFIG)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    gen = torch.Generator().manual_seed(CONFIG["seed"])                    # batch 표집 전용 RNG
    step, best = 0, float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False)                 # 직접 만든 파일만 이렇게 연다
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        gen.set_state(ckpt["rng"]["batch"])
        np.random.set_state(ckpt["rng"]["numpy"])
        random.setstate(ckpt["rng"]["python"])
        step, best = ckpt["step"], ckpt["best"]
        print(f"재개: step {step}, best 검증 MAE {best:.4f}")

    while step < args.steps:
        model.train()
        idx = torch.randint(0, len(Xn), (CONFIG["batch_size"],), generator=gen)
        loss = nn.functional.mse_loss(model(Xn[idx]), yn[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            val_mae = evaluate(model, Xv, yv, contract)
            if val_mae < best:
                best = val_mae
                save(args.out / "best.pt", model, opt, sched, gen, step, best, contract)
            print(f"step {step:4d}  train loss {loss.item():.4f}  검증 MAE {val_mae:.4f}")
        if step == args.stop_at:
            break
    save(args.out / "last.pt", model, opt, sched, gen, step, best, contract)
    print(f"저장: {args.out.name}/last.pt (step {step})")


if __name__ == "__main__":
    main()
