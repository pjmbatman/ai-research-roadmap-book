"""17.2 실습: in-place 연산·detach 누락·grad 누적으로 생기는 버그를 재현하고 고친다.

실행: uv run --project labs --group torch python labs/deep-learning-pytorch/tensors-and-autograd/autograd_bugs.py
CPU에서 1–2초. 각 버그마다 '버그'와 '수정' 두 버전을 나란히 실행한다.
"""
import weakref

import torch
from torch import nn


# ---------------------------------------------------------------- 1. detach 누락: 기록용 리스트가 그래프를 붙잡는다
class Holder:
    def __init__(self, t):
        self.t = t


class SavedTracker:
    """autograd가 backward용으로 저장한 텐서 중 아직 살아 있는 것의 총 바이트를 센다."""

    def __init__(self):
        self.live = 0

    def pack(self, t):
        holder = Holder(t.detach())                         # detach: 출력 텐서를 저장할 때 생기는 참조 순환을 피한다
        nbytes = t.numel() * t.element_size()
        self.live += nbytes
        weakref.finalize(holder, self._free, nbytes)       # 그래프가 해제되면 holder도 사라진다
        return holder

    def unpack(self, holder):
        return holder.t

    def _free(self, nbytes):
        self.live -= nbytes


def leak_demo(mode, steps=50):
    """평가 루프에서 손실을 합산한다. mode: "tensor"(버그) | "item" | "no_grad"."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 512), nn.Tanh(), nn.Linear(512, 1))
    tracker, total = SavedTracker(), 0.0
    with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
        for _ in range(steps):
            x, y = torch.randn(256, 64), torch.randn(256, 1)      # (B, D), (B, 1)
            if mode == "no_grad":
                with torch.no_grad():
                    total += nn.functional.mse_loss(model(x), y).item()
                continue
            loss = nn.functional.mse_loss(model(x), y)
            total = total + loss if mode == "tensor" else total + loss.item()   # 버그: 텐서째 합산
    return tracker.live / 2**20


# ---------------------------------------------------------------- 2. in-place 수정
def inplace_error_demo():
    a = torch.randn(4, requires_grad=True)
    s = torch.sigmoid(a)
    s.mul_(2)                                               # 버그: backward에 필요한 s를 덮어씀
    try:
        s.sum().backward()
        return "오류 없음"
    except RuntimeError as e:
        return "RuntimeError (" + str(e).split(":")[0] + ")"


def silent_data_demo(fix):
    """basic slicing은 view다. view를 제자리에서 정규화하면 원본 데이터가 바뀐다."""
    torch.manual_seed(0)
    data = torch.randn(64, 3) * 5 + 10                       # (N, D) 원본 데이터셋
    before = data.mean().item()
    for epoch in range(2):
        for i in range(0, 64, 16):
            xb = data[i:i + 16]                              # view
            if fix:
                xb = (xb - xb.mean(0)) / xb.std(0)           # 새 텐서
            else:
                xb -= xb.mean(0)                             # 버그: data가 바뀐다
                xb /= xb.std(0)
    return before, data.mean().item()


def data_attr_demo(fix):
    """.data로 수정하면 version counter를 우회해 오류 없이 틀린 gradient가 나온다."""
    a = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    s = torch.sigmoid(a)
    if fix:
        s = s * 2
    else:
        s.data.mul_(2)                                       # 버그: autograd 몰래 수정
    s.sum().backward()
    return a.grad


# ---------------------------------------------------------------- 3. grad 누적: zero_grad 누락
def accumulate_demo(zero_grad, steps=30):
    torch.manual_seed(0)
    X = torch.randn(128, 2)                                  # (N, D)
    y = X @ torch.tensor([2.0, -1.0]) + 0.5                  # (N,) 정확한 선형 관계
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for _ in range(steps):
        if zero_grad:
            opt.zero_grad()
        loss = nn.functional.mse_loss(model(X).squeeze(1), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main():
    torch.set_num_threads(1)
    print("[1] 평가 손실 합산과 그래프 메모리 (batch 50개 뒤 살아 있는 backward용 저장 텐서)")
    print(f"  total = total + loss            : {leak_demo('tensor'):6.1f} MiB")
    print(f"  total = total + loss.item()     : {leak_demo('item'):6.1f} MiB")
    print(f"  torch.no_grad() + loss.item()   : {leak_demo('no_grad'):6.1f} MiB")

    print("[2] in-place 수정")
    print("  sigmoid 출력을 mul_로 수정 후 backward :", inplace_error_demo())
    b0, b1 = silent_data_demo(fix=False)
    f0, f1 = silent_data_demo(fix=True)
    print(f"  view를 제자리 정규화: 데이터셋 평균 {b0:.3f} -> {abs(b1):.3f}  (오류 없이 원본이 바뀜)")
    print(f"  새 텐서로 정규화    : 데이터셋 평균 {f0:.3f} -> {f1:.3f}")
    print("  s.data.mul_(2) 후 a.grad :", data_attr_demo(fix=False).numpy().round(4), "(틀림, 오류 없음)")
    print("  s = s * 2      후 a.grad :", data_attr_demo(fix=True).numpy().round(4), "(맞음)")

    print("[3] zero_grad 누락 (선형회귀, SGD lr=0.1)")
    ok, bug = accumulate_demo(True), accumulate_demo(False)
    for k in [0, 5, 10, 20, 29]:
        print(f"  step {k:2d}  zero_grad 있음 {ok[k]:10.4f}   없음 {bug[k]:10.4f}")


if __name__ == "__main__":
    main()
