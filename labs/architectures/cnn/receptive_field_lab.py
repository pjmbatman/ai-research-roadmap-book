"""18.2 실습: 시계열 CNN의 receptive field가 필요한 맥락을 얼마나 덮는지와 성능의 관계.

데이터: 입력 x_t ~ N(0, 1) (외력), 출력 y_t = Σ_{j=0}^{J-1} h_j x_{t-j} (감쇠 진동자의 impulse response h로 걸러 낸 응답).
  h_j = exp(-j/τ) cos(2π j/P), τ = 12 step, P = 20 step, J = 96. 응답은 과거 수십 step의 외력에 의존한다.
모델: kernel 2, dilation 1, 2, 4, …인 causal dilated CNN(채널 32, ReLU, residual). 층 수 L이면 RF = 2^L.
비교: RF 밖에 있는 impulse response 에너지 비율  Σ_{j ≥ RF} h_j² / Σ_j h_j²  와 test 정규화 MSE.
x가 독립이므로, RF 밖의 외력이 만드는 응답 분산은 어떤 모델도 설명할 수 없다. 즉 그 비율이 정규화 MSE의 하한이다.

실행: uv run --project labs --group torch python labs/architectures/cnn/receptive_field_lab.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TAU, PERIOD, J = 12.0, 20.0, 96
T_SEQ, BATCH, STEPS = 512, 32, 1000


def impulse_response():
    j = np.arange(J)
    return np.exp(-j / TAU) * np.cos(2 * np.pi * j / PERIOD)          # (J,)


def make_batch(rng, h, n):
    x = rng.normal(size=(n, T_SEQ + J - 1))                            # 앞쪽 J-1 step은 워밍업
    y = np.stack([np.convolve(xi, h, mode="valid") for xi in x])       # (n, T_SEQ), y_t = Σ h_j x_{t-j}
    x = x[:, J - 1:]                                                   # (n, T_SEQ), y와 시점 정렬
    return torch.tensor(x[:, None], dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)


class CausalBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.pad = dilation                                            # kernel 2 → 왼쪽에 (k-1)·d 칸
        self.conv = nn.Conv1d(ch, ch, kernel_size=2, dilation=dilation)

    def forward(self, h):                                              # (B, C, T) → (B, C, T)
        return h + F.relu(self.conv(F.pad(h, (self.pad, 0))))          # 왼쪽만 padding: 미래를 보지 않음


class TCN(nn.Module):
    def __init__(self, n_layers, ch=32):
        super().__init__()
        self.inp = nn.Conv1d(1, ch, 1)
        self.blocks = nn.Sequential(*[CausalBlock(ch, 2 ** i) for i in range(n_layers)])
        self.out = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                                              # (B, 1, T) → (B, 1, T)
        return self.out(self.blocks(self.inp(x)))


def main():
    torch.set_num_threads(1)                                           # 작은 모델은 단일 스레드가 더 빠르다(집필 환경 기준)
    torch.manual_seed(0)
    h = impulse_response()
    energy = h ** 2
    rng_test = np.random.default_rng(123)
    x_te, y_te = make_batch(rng_test, h, 64)
    var_y = y_te[..., 128:].var().item()
    print(f"impulse response: τ = {TAU:.0f} step, 주기 {PERIOD:.0f} step, 길이 {J}")
    print("층 수   RF   RF 밖 에너지 비율   test 정규화 MSE")
    for n_layers in [2, 3, 4, 5, 6, 7]:
        rf = 2 ** n_layers
        rng = np.random.default_rng(n_layers)
        model = TCN(n_layers)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(STEPS):
            x, y = make_batch(rng, h, BATCH)
            loss = F.mse_loss(model(x)[..., 128:], y[..., 128:])       # 앞쪽 128 step은 zero padding 영향 구간이라 제외
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            mse = F.mse_loss(model(x_te)[..., 128:], y_te[..., 128:]).item()
        outside = energy[rf:].sum() / energy.sum()
        print(f"{n_layers:5d}  {rf:3d}   {outside:17.3f}   {mse / var_y:15.3f}")


if __name__ == "__main__":
    main()
