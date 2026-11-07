"""22.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 3·4는 PyTorch가 필요하고(exec="python", CPU의 gloo backend로 돈다), 나머지는 NumPy만 쓰는 브라우저 셀이다.
실행: uv run --project labs --group torch python labs/experiment-operations/distributed-training/cells.py
"""
# %% cell 1 — 데이터 병렬: worker별 gradient의 all-reduce 평균
import numpy as np

rng = np.random.default_rng(0)
N, D, W = 64, 3, 4                                 # 표본 수, 특성 수, worker 수
X = rng.normal(size=(N, D))                        # (N, D)
y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=N)   # (N,)
w = np.zeros(D)                                    # 모든 worker가 같은 파라미터를 들고 있다

def grad(Xb, yb):                                  # 평균 MSE의 gradient, (D,)
    return 2 * Xb.T @ (Xb @ w - yb) / len(yb)

g_full = grad(X, y)                                # 한 장치가 batch 64 전체를 본다
local = [grad(Xs, ys) for Xs, ys in zip(np.split(X, W), np.split(y, W))]   # worker마다 16개
g_allreduce = np.sum(local, axis=0) / W            # all-reduce(합) 뒤 W로 나눈다
print("worker별 gradient가 서로 다른가:", not np.allclose(local[0], local[1]))
print("all-reduce 평균 == 전체 batch gradient:", np.allclose(g_allreduce, g_full))

sizes = [40, 8, 8, 8]                              # 표본이 불균등하게 나뉘면
parts = np.split(np.arange(N), np.cumsum(sizes)[:-1])
g_uneven = np.mean([grad(X[p], y[p]) for p in parts], axis=0)
print("불균등 분할에서 단순 평균 == 전체 batch gradient:", np.allclose(g_uneven, g_full))

# %% cell 2 — 분할(sharding)에 따른 GPU당 정적 메모리
P, W = 7e9, 8                                      # 파라미터 수, GPU 수 (fp32 + Adam: 4 + 4 + 8 byte)
schemes = {                                        # GPU 하나가 들고 있는 byte/파라미터: (파라미터, gradient, optimizer 상태)
    "DDP (모두 복제)":                   (4, 4, 8),
    "ZeRO-1 (optimizer 상태 분할)":      (4, 4, 8 / W),
    "ZeRO-2 (+ gradient 분할)":          (4, 4 / W, 8 / W),
    "ZeRO-3 / FSDP (+ 파라미터 분할)":   (4 / W, 4 / W, 8 / W),
}
for name, b in schemes.items():
    print(f"{name:26s} {sum(b):5.2f} byte/파라미터 → GPU당 {P * sum(b) / 2**30:6.1f} GiB")

# %% cell 3 — 실제 DDP: gloo backend, 프로세스 2개
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

def worker(rank, world, init_file, X, y, out):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = DDP(nn.Linear(3, 1))                   # 생성할 때 rank 0의 파라미터를 모든 rank에 방송한다
    xs, ys = X.chunk(world)[rank], y.chunk(world)[rank]          # rank마다 다른 절반, (4, 3)
    nn.functional.mse_loss(model(xs), ys).backward()             # backward 중에 gradient를 all-reduce(평균)
    out.put((rank, model.module.weight.grad.flatten().tolist()))
    dist.destroy_process_group()

torch.manual_seed(1)
Xt, yt = torch.randn(8, 3), torch.randn(8, 1)      # (8, 3), (8, 1)
ctx = mp.get_context("fork")
out, init_file = ctx.SimpleQueue(), tempfile.mktemp()    # 두 프로세스가 만나는 rendezvous 파일 경로
procs = [ctx.Process(target=worker, args=(r, 2, init_file, Xt, yt, out)) for r in range(2)]
for p in procs: p.start()
grads = dict(out.get() for _ in procs)
for p in procs: p.join()

torch.manual_seed(0)
ref = nn.Linear(3, 1)                              # 한 프로세스가 8개 전체를 본다
nn.functional.mse_loss(ref(Xt), yt).backward()
print("rank 0 gradient:", [round(v, 4) for v in grads[0]])
print("rank 1 gradient:", [round(v, 4) for v in grads[1]])
print("단일 프로세스 :", [round(v, 4) for v in ref.weight.grad.flatten().tolist()])

# %% cell 4 — DistributedSampler: rank별 표본과 epoch별 순서
from torch.utils.data.distributed import DistributedSampler

data = list(range(10))                             # 표본 10개, rank 4개
for epoch in [0, 1]:
    parts = []
    for r in range(4):
        s = DistributedSampler(data, num_replicas=4, rank=r, shuffle=True, seed=0)
        s.set_epoch(epoch)                         # 빠뜨리면 매 epoch 같은 순서
        parts.append(list(s))
    print(f"epoch {epoch}: {parts}")
flat = sum(parts, [])
print(f"전체 {len(flat)}개, 고유 {len(set(flat))}개 → 중복 {len(flat) - len(set(flat))}개")

# %% cell 5 — GPU 수에 따라 바뀌는 학습 설정
N, per_gpu, base_lr, epochs = 50_000, 32, 0.1, 90  # 기준: GPU 1개, batch 32에서 튜닝한 학습률 0.1
print(" GPU  유효 batch  step/epoch  총 step  학습률(선형)  학습률(제곱근)")
for g in [1, 2, 4, 8, 32]:
    B = per_gpu * g
    steps = int(np.ceil(N / B))
    print(f"{g:4d} {B:10d} {steps:11d} {steps * epochs:8d} {base_lr * g:12.2f} {base_lr * np.sqrt(g):14.3f}")

# %% cell 6 — 분산 평가: rank 0만 보고하면 생기는 일
rng = np.random.default_rng(1)
N, W = 1001, 4                                     # 검증 표본 1001개, rank 4개
is_correct = rng.random(N) < 0.8                   # 표본별 정답 여부, (N,)
total = int(np.ceil(N / W)) * W                    # DistributedSampler는 W의 배수로 맞춘다
idx = np.concatenate([np.arange(N), np.arange(total - N)])   # 앞쪽 표본을 반복해 채운다(shuffle 없음)
shards = [idx[r::W] for r in range(W)]
print("rank별 정확도:", [f"{is_correct[s].mean():.4f}" for s in shards])
print(f"rank 0만 보고        {is_correct[shards[0]].mean():.4f}")
print(f"전 rank 합산(중복 포함) {np.mean(np.concatenate([is_correct[s] for s in shards])):.4f}  ({total}개)")
print(f"실제 정확도           {is_correct.mean():.4f}  ({N}개)")

# %% cell 7 — 통신 비용 추정: ring all-reduce
P = 350e6                                          # 파라미터 수
grad_bytes = P * 4                                 # fp32 gradient 한 벌, byte
links_gbps = {"연결 A (가정 10 GB/s)": 10e9, "연결 B (가정 100 GB/s)": 100e9}
compute_s = 0.30                                   # GPU당 forward+backward 시간 (가정)
for W in [2, 8, 64]:
    sent = 2 * (W - 1) / W * grad_bytes            # GPU 하나가 보내는(=받는) byte
    for name, bw in links_gbps.items():
        t = sent / bw
        print(f"W={W:2d} {name}: 보내는 양 {sent / 1e9:5.2f} GB, 통신 {t * 1e3:5.0f} ms = 계산의 {t / compute_s:4.0%}")
