"""17.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 2·4는 NumPy만 쓰는 브라우저 셀이고(exec="browser"), 나머지는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/deep-learning-pytorch/datasets-and-dataloaders/cells.py
"""
# %% cell 1 — map-style Dataset과 sample contract
import torch
from torch.utils.data import DataLoader, Dataset

class StationDataset(Dataset):
    """관측소별 기온 창(window) 하나를 sample 하나로 돌려준다."""
    def __init__(self, temps_k, window, station_ids):
        self.temps_k, self.window, self.ids = temps_k, window, station_ids   # (S, T_total) K

    def __len__(self):
        return self.temps_k.shape[0] * (self.temps_k.shape[1] - self.window)

    def __getitem__(self, i):
        s, t = divmod(i, self.temps_k.shape[1] - self.window)
        x = self.temps_k[s, t:t + self.window]             # (T,) 입력 창, K
        return {"x": x[:, None].float(),                   # (T, C=1) float32
                "y": self.temps_k[s, t + self.window].float(),   # () 다음 시각 기온, K
                "station": self.ids[s]}                    # metadata (문자열)

torch.manual_seed(0)
ds = StationDataset(280 + 5 * torch.randn(3, 50, dtype=torch.float64), window=24,
                    station_ids=["A", "B", "C"])
batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
print("len(ds) =", len(ds))
print({k: (tuple(v.shape), v.dtype) if torch.is_tensor(v) else v for k, v in batch.items()})

# %% cell 2 — padding과 mask: padding이 손실에 섞이면
import numpy as np

rng = np.random.default_rng(0)
lengths = [3, 7, 5]                                         # 가변 길이 시계열 3개
seqs = [rng.normal(loc=2.0, size=n) for n in lengths]
T = max(lengths)
x = np.zeros((len(seqs), T))                                # (B, T) 0으로 padding
mask = np.zeros((len(seqs), T), dtype=bool)                 # (B, T) 실제 값이면 True
for i, s in enumerate(seqs):
    x[i, :len(s)], mask[i, :len(s)] = s, True
pred = np.full_like(x, 2.0)                                 # 모든 시각에서 2.0을 예측
err = (pred - x) ** 2
print("padding 포함 MSE :", round(err.mean(), 4))
print("mask 적용 MSE    :", round(err[mask].mean(), 4))
print("표본별로 평균 후 평균:", round(np.mean([e[m].mean() for e, m in zip(err, mask)]), 4))
print("padding 비율     :", round(1 - mask.mean(), 3))

# %% cell 3 — collate_fn으로 가변 길이 batch 만들기
from torch.nn.utils.rnn import pad_sequence

def make_seq(n, seed):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(n, 2, generator=g), "y": torch.tensor(float(n))}   # x: (T_i, C)

samples = [make_seq(n, i) for i, n in enumerate([3, 7, 5])]
try:
    torch.utils.data.default_collate(samples)
except RuntimeError as e:
    print("default_collate -> RuntimeError:", "equal size" in str(e))

def pad_collate(samples):
    xs = [s["x"] for s in samples]
    lengths = torch.tensor([len(x) for x in xs])                         # (B,)
    x = pad_sequence(xs, batch_first=True, padding_value=0.0)            # (B, T_max, C)
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]          # (B, T_max)
    return {"x": x, "mask": mask, "lengths": lengths, "y": torch.stack([s["y"] for s in samples])}

b = pad_collate(samples)
print({k: tuple(v.shape) for k, v in b.items()})
print(b["mask"].int())

# %% cell 4 — sampler: epoch마다 다른 shuffle, 불균형 데이터의 가중 추출
rng = np.random.default_rng(0)
N, B = 10, 4
for epoch in range(2):
    perm = np.random.default_rng([0, epoch]).permutation(N)   # (시드, epoch)에서 결정
    batches = [perm[i:i + B].tolist() for i in range(0, N, B)]
    print(f"epoch {epoch}:", batches, " drop_last면 마지막 batch 버림")

labels = np.array([0] * 95 + [1] * 5)                       # 클래스 1이 5%
w = 1.0 / np.bincount(labels)[labels]                       # 표본 가중치 = 1 / 클래스 빈도
idx = rng.choice(len(labels), size=2000, replace=True, p=w / w.sum())
print("균일 추출 시 클래스 1 비율: 0.05,  가중 추출:", round(labels[idx].mean(), 3))
print("가중 추출에서 서로 다른 클래스 1 표본 수:", len(set(idx[labels[idx] == 1])))

# %% cell 5 — worker마다 같은 난수: Dataset 안의 Generator
import pickle
import numpy as np

class NoisyDataset(Dataset):
    def __init__(self):
        self.rng = np.random.default_rng(0)                  # __init__에서 만든 Generator
    def __len__(self):
        return 100
    def __getitem__(self, i):
        return self.rng.normal()                             # 증강용 잡음이라고 하자

def worker_init_fn(worker_id):                               # DataLoader(worker_init_fn=...)
    info = torch.utils.data.get_worker_info()                # worker 안에서만 None이 아니다
    info.dataset.rng = np.random.default_rng(info.seed)      # info.seed는 worker마다 다르다

ds = NoisyDataset()
workers = [pickle.loads(pickle.dumps(ds)) for _ in range(2)]   # worker마다 복사본이 간다
print("복사본 그대로      :", [round(w[0], 4) for w in workers])
for worker_id, w in enumerate(workers):                      # worker_init_fn이 하는 일을 손으로
    w.rng = np.random.default_rng(1234 + worker_id)          # seed = base_seed + worker_id
print("worker별로 다시 시드:", [round(w[0], 4) for w in workers])
