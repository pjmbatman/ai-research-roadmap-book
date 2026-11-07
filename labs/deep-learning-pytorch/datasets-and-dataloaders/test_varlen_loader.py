"""17.4 실습 코드(varlen_loader.py)의 데이터 로더 단독 테스트. PyTorch가 없으면 건너뛴다."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location("varlen_loader", Path(__file__).parent / "varlen_loader.py")
vl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vl)


@pytest.fixture(scope="module")
def ds():
    return vl.SensorSeqDataset(n=16, seed=3)


def test_sample_contract(ds):
    s = ds[0]
    assert set(s) == {"x", "y", "length", "meta"}
    assert s["x"].dtype == torch.float32 and s["x"].shape == (int(s["length"]), len(vl.CHANNELS))
    assert s["y"].shape == () and s["length"].dtype == torch.int64
    assert 250 < float(s["y"]) < 350                       # 단위: K


def test_getitem_does_not_share_memory(ds):
    s = ds[1]
    s["x"][0, 0] = -1.0
    assert ds.series[1][0, 0] != -1.0


def test_pad_collate_shapes_and_mask(ds):
    samples = [ds[i] for i in range(5)]
    b = vl.pad_collate(samples, pad_value=-7.0)
    T = max(int(s["length"]) for s in samples)
    assert b["x"].shape == (5, T, 2) and b["mask"].shape == (5, T)
    assert b["mask"].sum(1).tolist() == [int(s["length"]) for s in samples]
    for i, s in enumerate(samples):
        n = int(s["length"])
        assert torch.equal(b["x"][i, :n], s["x"])           # 실제 값은 그대로
        assert torch.all(b["x"][i, n:] == -7.0)             # padding은 지정한 값
    assert [m["run_id"] for m in b["meta"]] == [s["meta"]["run_id"] for s in samples]


def test_masked_mean_ignores_padding(ds):
    samples = [ds[i] for i in range(6)]
    for pad in (0.0, 1e6):                                  # padding 값을 바꿔도 결과가 같아야 한다
        b = vl.pad_collate(samples, pad_value=pad)
        got = vl.masked_mean(b["x"][..., 0], b["mask"])
        want = torch.stack([s["x"][:, 0].mean() for s in samples])
        assert torch.allclose(got, want, rtol=1e-5)
    assert torch.allclose(got, b["y"], rtol=1e-5)


def test_loader_batches_cover_dataset(ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=True, collate_fn=vl.pad_collate,
                                         generator=torch.Generator().manual_seed(0))
    ids = [m["run_id"] for b in loader for m in b["meta"]]
    assert sorted(ids) == sorted(f"run-{i:03d}" for i in range(len(ds)))   # epoch마다 모든 표본 한 번씩


def test_workers_get_different_noise():
    vals = vl.first_noise_per_worker(vl.reseed_worker)
    assert vals[0] != vals[1], "worker마다 다른 난수 흐름이어야 한다"
    dup = vl.first_noise_per_worker(None)
    assert dup[0] == dup[1], "Generator를 다시 시드하지 않으면 worker들이 같은 난수를 쓴다"
