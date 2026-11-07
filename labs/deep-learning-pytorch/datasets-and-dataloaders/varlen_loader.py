"""17.4 실습: 가변 길이 시계열을 padding·mask로 batch화하는 Dataset·collate_fn과 그 검증.

실행: uv run --project labs --group torch python labs/deep-learning-pytorch/datasets-and-dataloaders/varlen_loader.py
테스트: cd labs && uv run --group torch pytest -q deep-learning-pytorch/datasets-and-dataloaders/test_varlen_loader.py
그림: labs/deep-learning-pytorch/datasets-and-dataloaders/outputs/batch.png
"""
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, get_worker_info

HERE = Path(__file__).parent
CHANNELS = ("temp_k", "pressure_kpa")                      # 채널 이름과 단위


class SensorSeqDataset(Dataset):
    """길이가 다른 센서 시계열. sample contract:

    x      (T_i, C) float32   입력 시계열, 채널 순서와 단위는 CHANNELS
    y      ()       float32   시계열 전체의 평균 온도(K), 예측 목표
    length ()       int64     실제 길이 T_i
    meta   dict               {"run_id": str}
    """

    def __init__(self, n=64, min_len=20, max_len=100, seed=0, noise_std=0.0):
        rng = np.random.default_rng(seed)
        self.lengths = rng.integers(min_len, max_len + 1, size=n)
        self.series = []
        for i, T in enumerate(self.lengths):
            t = np.arange(T)
            temp = 300 + 5 * np.sin(2 * np.pi * t / 40 + i) + rng.normal(0, 0.5, T)   # K
            pres = 101.3 + 0.2 * rng.normal(size=T)                                     # kPa
            self.series.append(np.stack([temp, pres], axis=1).astype(np.float32))      # (T, C)
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed + 1)          # 증강용. worker마다 다시 시드해야 한다

    def __len__(self):
        return len(self.series)

    def __getitem__(self, i):
        x = torch.from_numpy(self.series[i].copy())         # (T, C) 원본을 공유하지 않게 복사
        if self.noise_std > 0:
            x = x + torch.from_numpy(self.rng.normal(0, self.noise_std, x.shape).astype(np.float32))
        return {"x": x, "y": torch.tensor(self.series[i][:, 0].mean()),
                "length": torch.tensor(len(x)), "meta": {"run_id": f"run-{i:03d}"}}


def pad_collate(samples, pad_value=0.0):
    """sample 목록을 batch dict로. x: (B, T_max, C), mask: (B, T_max) bool."""
    lengths = torch.stack([s["length"] for s in samples])                       # (B,)
    x = pad_sequence([s["x"] for s in samples], batch_first=True, padding_value=pad_value)
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]                 # (B, T_max)
    return {"x": x, "mask": mask, "lengths": lengths,
            "y": torch.stack([s["y"] for s in samples]),
            "meta": [s["meta"] for s in samples]}


def masked_mean(values, mask):
    """values: (B, T) 또는 (B, T, C), mask: (B, T). padding을 뺀 표본별 평균 (B,) 또는 (B, C)."""
    m = mask.to(values.dtype)
    if values.dim() == 3:
        m = m[..., None]
    return (values * m).sum(1) / m.sum(1)


def reseed_worker(worker_id):
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed)


def first_noise_per_worker(worker_init_fn):
    """worker 2개로 batch 크기 1씩 읽어, 각 worker가 처음 만든 잡음 값을 모은다."""
    ds = SensorSeqDataset(n=4, min_len=30, max_len=30, noise_std=1.0)   # 길이를 같게 해 난수 소비량을 맞춘다
    loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=pad_collate,
                        worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(0))
    out = []
    for b in loader:                                         # worker 0, 1이 번갈아 batch를 낸다
        i = int(b["meta"][0]["run_id"][-3:])
        out.append(round(float(b["x"][0, 0, 0] - ds.series[i][0, 0]), 4))
    return out


def plot_batch(batch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B = batch["x"].shape[0]
    fig, axes = plt.subplots(B, 1, figsize=(7, 1.6 * B), sharex=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        x, m = batch["x"][i, :, 0], batch["mask"][i]
        ax.plot(np.where(m, x, np.nan), lw=1, label="temp_k (valid)")
        ax.plot(np.where(~m, x, np.nan), lw=1, color="0.6", ls="--", label="padding")
        ax.set_ylabel(batch["meta"][i]["run_id"], fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("time index")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    ds = SensorSeqDataset()
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=pad_collate,
                        generator=torch.Generator().manual_seed(0))
    batch = next(iter(loader))
    print("[1] batch 하나")
    for k, v in batch.items():
        print(f"  {k:8s}", tuple(v.shape) if torch.is_tensor(v) else v)
    print("  lengths:", batch["lengths"].tolist(), " mask 행 합:", batch["mask"].sum(1).tolist())
    naive = batch["x"][..., 0].mean(1)
    masked = masked_mean(batch["x"][..., 0], batch["mask"])
    fmt = lambda t: [round(v, 1) for v in t.tolist()]
    print("  padding 포함 평균 온도(K):", fmt(naive))
    print("  mask 적용 평균 온도(K)   :", fmt(masked))
    print("  목표 y(K)               :", fmt(batch["y"]))
    plot_batch(batch, HERE / "outputs/batch.png")
    print("  그림 저장: outputs/batch.png")

    print("[2] 각 batch 첫 원소에 더해진 잡음 (batch 0·2는 worker 0, 1·3은 worker 1)")
    print("  worker_init_fn 없음 :", first_noise_per_worker(None))
    print("  reseed_worker 사용  :", first_noise_per_worker(reseed_worker))


if __name__ == "__main__":
    main()
