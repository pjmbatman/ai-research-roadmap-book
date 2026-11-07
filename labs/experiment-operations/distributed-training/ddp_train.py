"""22.6 실습: 단일 프로세스 학습을 DDP로 바꾸고, 프로세스 수에 따라 결과가 같은지 비교한다.

CPU에서 gloo backend로 돈다(GPU가 있으면 --backend nccl, 프로세스마다 GPU 하나).
전역 batch를 고정하고 rank당 batch = 전역 batch / world size로 나누므로, 프로세스 수가 바뀌어도
같은 step에 같은 표본을 보고 같은 학습을 해야 한다.

실행 (저장소 루트에서, 집필 시점 예시):
    uv run --project labs --group torch torchrun --standalone --nproc_per_node=1 \
        labs/experiment-operations/distributed-training/ddp_train.py
    uv run --project labs --group torch torchrun --standalone --nproc_per_node=2 \
        labs/experiment-operations/distributed-training/ddp_train.py
    # rank당 batch를 그대로 둔 채 프로세스만 늘린 경우(유효 batch 2배)
    uv run --project labs --group torch torchrun --standalone --nproc_per_node=2 \
        labs/experiment-operations/distributed-training/ddp_train.py --per-rank-batch 64
"""
import argparse
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

N, D, GLOBAL_BATCH, EPOCHS, LR = 512, 8, 64, 5, 0.05


def make_data():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(N, D, generator=g)                                  # (N, D)
    w_true = torch.linspace(-1.0, 1.0, D)
    y = torch.tanh(X @ w_true).unsqueeze(1) + 0.05 * torch.randn(N, 1, generator=g)   # (N, 1)
    return TensorDataset(X, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--per-rank-batch", type=int, default=None,
                        help="지정하면 rank당 batch를 이 값으로 고정한다(유효 batch가 world size에 비례해 커진다)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group(args.backend)                  # torchrun이 RANK·WORLD_SIZE·MASTER_ADDR를 넣어 준다
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.backend == "nccl" else torch.device("cpu")

    data = make_data()
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=0)
    per_rank = args.per_rank_batch or GLOBAL_BATCH // world
    loader = DataLoader(data, batch_size=per_rank, sampler=sampler)
    if rank == 0:
        print(f"world {world}  rank당 batch {per_rank}  유효 batch {per_rank * world}")
    torch.manual_seed(0)                                   # 모든 rank가 같은 초기값 (DDP도 rank 0 값을 방송한다)
    model = DDP(nn.Sequential(nn.Linear(D, 32), nn.Tanh(), nn.Linear(32, 1)).to(device))
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    steps = 0
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)                           # epoch마다 다른 순서, 모든 rank가 같은 순서를 공유
        loss_sum = torch.zeros(2, device=device)           # [손실 합, 표본 수]
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()                                # DDP가 여기서 gradient를 all-reduce(평균)한다
            opt.step()
            steps += 1
            loss_sum += torch.stack([loss.detach() * len(xb), torch.tensor(float(len(xb)), device=device)])
        dist.all_reduce(loss_sum)                          # 모든 rank의 합 → 전체 평균
        if rank == 0:
            print(f"world {world}  epoch {epoch + 1}  step {steps:3d}  학습 손실 {(loss_sum[0] / loss_sum[1]).item():.6f}")
    if rank == 0:
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        print(f"world {world}  가중치 합 {flat.sum().item():.6f}  노름 {flat.norm().item():.6f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
