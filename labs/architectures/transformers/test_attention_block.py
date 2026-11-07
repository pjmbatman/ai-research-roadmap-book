"""18.4 실습: 직접 구현한 attention 블록의 shape·mask·gradient를 테스트로 검증한다.

PyTorch가 없으면 건너뛴다. 실행: cd labs && uv run --group torch pytest -q architectures/transformers/test_attention_block.py
"""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location("attention_block", Path(__file__).parent / "attention_block.py")
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

B, T, D, H, D_FF = 2, 6, 16, 4, 32


@pytest.fixture
def block():
    torch.manual_seed(0)
    return ab.Block(D, H, D_FF).eval()


def test_shape(block):
    assert block(torch.randn(B, T, D)).shape == (B, T, D)


def test_matches_pytorch_layer(block):
    ref = ab.copy_to_reference(block, D, H, D_FF)
    x = torch.randn(B, T, D)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    with torch.no_grad():
        assert torch.allclose(block(x), ref(x), atol=1e-5)
        assert torch.allclose(block(x, causal), ref(x, src_mask=~causal), atol=1e-5)


def test_causal_mask_blocks_future(block):
    x = torch.randn(B, T, D)
    x2 = x.clone()
    x2[:, 4] += 3.0
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    with torch.no_grad():
        y, y2 = block(x, causal), block(x2, causal)
    assert torch.allclose(y[:, :4], y2[:, :4]), "미래 토큰이 과거 출력에 영향을 주면 안 된다"
    assert not torch.allclose(y[:, 4:], y2[:, 4:])


def test_padding_mask_blocks_pad(block):
    lengths = torch.tensor([6, 3])
    key_ok = torch.arange(T)[None, :] < lengths[:, None]           # (B, T)
    mask = key_ok[:, None, None, :].expand(B, 1, T, T)              # (B, 1, T, T), head 축에 broadcast
    x = torch.randn(B, T, D)
    x2 = x.clone()
    x2[1, 3:] = torch.randn(3, D) * 10
    with torch.no_grad():
        y, y2 = block(x, mask), block(x2, mask)
    assert torch.allclose(y[1, :3], y2[1, :3], atol=1e-6), "PAD 값이 실제 토큰 출력에 섞이면 안 된다"
    with torch.no_grad():
        y_nomask, y2_nomask = block(x), block(x2)
    assert not torch.allclose(y_nomask[1, :3], y2_nomask[1, :3]), "mask가 없으면 PAD가 섞이는 것이 정상"


def test_gradient_matches_finite_difference():
    torch.manual_seed(1)
    blk = ab.Block(8, 2, 16).double()
    x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    assert torch.autograd.gradcheck(lambda z: blk(z, causal).sum(), (x,), eps=1e-6, atol=1e-5)


def test_gradient_reaches_all_parameters(block):
    block.train()
    loss = block(torch.randn(B, T, D)).pow(2).mean()
    loss.backward()
    for name, p in block.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name}에 gradient가 흐르지 않는다"
