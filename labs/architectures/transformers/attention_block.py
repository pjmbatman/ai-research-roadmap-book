"""18.4 실습: 작은 Transformer 블록을 직접 구현하고 PyTorch 구현과 비교한다.

pre-norm encoder 블록:  h = x + MHA(LN(x)),  y = h + FFN(LN(h))
같은 가중치를 nn.TransformerEncoderLayer(norm_first=True, dropout=0)에 복사해 출력이 같은지 확인한다.
shape·mask·gradient 검증은 test_attention_block.py가 한다.

실행: uv run --project labs --group torch python labs/architectures/transformers/attention_block.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):                      # x (B, T, D), mask (T, T) 또는 (B, 1, T, T), True = 볼 수 있음
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)   # 각 (B, H, T, Dh)
        s = q @ k.transpose(-1, -2) / self.dh ** 0.5      # (B, H, T, T)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        o = s.softmax(-1) @ v                             # (B, H, T, Dh)
        return self.out(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask=None):                      # (B, T, D) → (B, T, D)
        x = x + self.attn(self.ln1(x), mask)              # pre-norm: 정규화는 residual 가지 안에
        return x + self.ff(self.ln2(x))


def copy_to_reference(block, d_model, n_heads, d_ff):
    """같은 가중치를 가진 nn.TransformerEncoderLayer를 만든다."""
    ref = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, activation="relu",
                                     batch_first=True, norm_first=True)
    with torch.no_grad():
        ref.self_attn.in_proj_weight.copy_(block.attn.qkv.weight)
        ref.self_attn.in_proj_bias.copy_(block.attn.qkv.bias)
        ref.self_attn.out_proj.weight.copy_(block.attn.out.weight)
        ref.self_attn.out_proj.bias.copy_(block.attn.out.bias)
        ref.linear1.load_state_dict(block.ff[0].state_dict())
        ref.linear2.load_state_dict(block.ff[2].state_dict())
        ref.norm1.load_state_dict(block.ln1.state_dict())
        ref.norm2.load_state_dict(block.ln2.state_dict())
    return ref.eval()


def main():
    torch.manual_seed(0)
    B, T, D, H, D_ff = 3, 9, 32, 4, 64
    block = Block(D, H, D_ff).eval()
    ref = copy_to_reference(block, D, H, D_ff)
    x = torch.randn(B, T, D)
    causal = torch.ones(T, T, dtype=torch.bool).tril()   # 우리 규칙: True = 볼 수 있음
    with torch.no_grad():
        y, y_ref = block(x), ref(x)
        yc, yc_ref = block(x, causal), ref(x, src_mask=~causal)   # PyTorch bool mask: True = 가림
    print(f"블록 파라미터 수: {sum(p.numel() for p in block.parameters())}"
          f"  (PyTorch 구현: {sum(p.numel() for p in ref.parameters())})")
    print(f"mask 없음  출력 shape {tuple(y.shape)}  최대 차 < 1e-5: {(y - y_ref).abs().max().item() < 1e-5}")
    print(f"causal     출력 shape {tuple(yc.shape)}  최대 차 < 1e-5: {(yc - yc_ref).abs().max().item() < 1e-5}")


if __name__ == "__main__":
    main()
