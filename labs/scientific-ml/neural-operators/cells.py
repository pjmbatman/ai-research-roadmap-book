"""24.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

cell 5는 PyTorch가 필요하다(exec="python").
실행: uv run --project labs --group torch python labs/scientific-ml/neural-operators/cells.py
"""
# %% cell 1 — 입력 함수와 정답 연산자: 주기 영역의 열방정식과 이류
import numpy as np

rng = np.random.default_rng(0)
K_IN = 12                                               # 입력 함수가 가진 최고 파수

def sample_functions(n, K=K_IN, decay=1.5):            # 함수 n개를 Fourier 계수로 정의, (n, K) 복소수
    k = np.arange(1, K + 1)
    return (rng.normal(size=(n, K)) + 1j * rng.normal(size=(n, K))) / k**decay

def on_grid(coef, N):                                   # 계수 → 격자 [0, 1)의 N점 값, (n, N)
    k = np.arange(1, coef.shape[1] + 1)
    x = np.arange(N) / N
    return 2 * np.real(coef @ np.exp(2j * np.pi * np.outer(k, x)))

def heat(coef, T=0.002):                                # u_t = u_xx를 T까지: 모드마다 exp(-(2πk)² T)
    k = np.arange(1, coef.shape[1] + 1)
    return coef * np.exp(-(2 * np.pi * k) ** 2 * T)

def advect(coef, shift=0.3):                            # u_t + u_x = 0을 0.3까지: 모드마다 위상 회전
    k = np.arange(1, coef.shape[1] + 1)
    return coef * np.exp(-2j * np.pi * k * shift)

def rel_err(pred, true):                                # 상대 L2 오차
    return np.linalg.norm(pred - true) / np.linalg.norm(true)

fmt = lambda e: "< 1e-12" if e < 1e-12 else f"{e:.1e}"  # 반올림 수준의 값은 판정으로
a_train, a_test = sample_functions(200), sample_functions(50)   # 학습 입력 200개, 시험 입력 50개
print("격자 64점의 입력 배열:", on_grid(a_train, 64).shape, " 격자 256점:", on_grid(a_train, 256).shape)
print("같은 함수, x = 0.25의 값 (64점, 256점):", on_grid(a_train[:1], 64)[0, 16].round(6), on_grid(a_train[:1], 256)[0, 64].round(6))

# %% cell 2 — DeepONet의 구조: branch(센서 값) × trunk(출력 위치)
m, p = 32, 25                                           # 센서 수, basis 함수 수
sensors = np.arange(m) / m                              # 고정된 센서 위치, (32,)

def trunk(y):                                           # 출력 위치 y에서의 basis 값, (len(y), p)
    k = np.arange(1, (p - 1) // 2 + 1)
    return np.hstack([np.ones((len(y), 1)), np.cos(2 * np.pi * np.outer(y, k)), np.sin(2 * np.pi * np.outer(y, k))])

y_fit = np.arange(512) / 512
coef = np.linalg.lstsq(trunk(y_fit), on_grid(heat(a_train), 512).T, rcond=None)[0].T   # 출력 함수의 basis 계수, (200, p)
B = np.linalg.lstsq(on_grid(a_train, m), coef, rcond=None)[0]   # 선형 branch: 센서 값 → 계수, (32, p)

def deeponet(u_at_sensors, y):                          # G(u)(y) = Σ_k branch_k(u) · trunk_k(y), (n, len(y))
    return (u_at_sensors @ B) @ trunk(y).T

y_query = rng.uniform(0, 1, 1000)                       # 학습에 쓰지 않은 임의의 출력 위치
true = 2 * np.real(heat(a_test) @ np.exp(2j * np.pi * np.outer(np.arange(1, K_IN + 1), y_query)))
x50 = np.arange(50) / 50
u50 = on_grid(a_test, 50)                               # 다른 장비: 50점 격자로 측정한 입력
shifted = 2 * np.real(a_test @ np.exp(2j * np.pi * np.outer(np.arange(1, K_IN + 1), sensors + 0.5 / m)))
interp = np.array([np.interp(sensors, x50, u, period=1.0) for u in u50])
print(f"정해진 센서 값, 임의 위치 1000곳     상대 오차 {fmt(rel_err(deeponet(on_grid(a_test, m), y_query), true))}")
print(f"센서가 반 칸 어긋난 입력             상대 오차 {rel_err(deeponet(shifted, y_query), true):.1e}")
print(f"50점 입력을 센서 위치로 선형 보간    상대 오차 {rel_err(deeponet(interp, y_query), true):.1e}")

# %% cell 3 — 스펙트럼 합성곱 층과 해상도
def spectral_conv(u, w):                                # u: (n, N) 격자 값, w: (K,) 복소 가중치 → (n, N)
    N, K = u.shape[1], w.size
    u_hat = np.fft.rfft(u, axis=1) / N                  # 격자 크기와 무관한 Fourier 계수, (n, N//2+1)
    out_hat = np.zeros_like(u_hat)
    out_hat[:, 1:K + 1] = w * u_hat[:, 1:K + 1]         # 낮은 K개 모드에만 가중치, 나머지는 버림
    return np.fft.irfft(out_hat * N, n=N, axis=1)

def stencil_conv(u, kernel=(0.25, 0.5, 0.25)):         # 격자 칸 단위의 3점 합성곱 (보통의 CNN 층)
    return kernel[0] * np.roll(u, 1, axis=1) + kernel[1] * u + kernel[2] * np.roll(u, -1, axis=1)

w = rng.normal(size=16) + 1j * rng.normal(size=16)      # 무작위 스펙트럼 가중치, 모드 16개
f = a_train[:5]                                         # 함수 5개
ref_s, ref_c = spectral_conv(on_grid(f, 32), w), stencil_conv(on_grid(f, 32))
for N in [64, 128, 256]:
    step = N // 32                                      # 32점 격자와 겹치는 점만 비교
    d_s = np.abs(spectral_conv(on_grid(f, N), w)[:, ::step] - ref_s).max()
    d_c = np.abs(stencil_conv(on_grid(f, N))[:, ::step] - ref_c).max()
    print(f"N = {N:3d}  32점 결과와의 최대 차이: 스펙트럼 합성곱 {'< 1e-12' if d_s < 1e-12 else f'{d_s:.1e}'}"
          f"   3점 합성곱 {d_c:.2f}")

# %% cell 4 — 스펙트럼 가중치를 데이터로 학습하고 다른 해상도에서 평가
def fit_spectral(u_in, u_out, K):                       # 모드마다 최소제곱: w_k = Σ out·conj(in) / Σ |in|²
    N = u_in.shape[1]
    a, b = np.fft.rfft(u_in, axis=1)[:, 1:K + 1] / N, np.fft.rfft(u_out, axis=1)[:, 1:K + 1] / N
    return (b * a.conj()).sum(0) / (np.abs(a) ** 2).sum(0)

a_rough = sample_functions(50, K=30, decay=1.0)         # 파수 30까지, 고주파가 더 큰 함수
for name, op in [("열방정식", heat), ("이류    ", advect)]:
    w = fit_spectral(on_grid(a_train, 64), on_grid(op(a_train), 64), K=K_IN)  # 해상도 64에서 모드 12개 학습
    msg = []
    for N in [64, 256]:
        smooth = rel_err(spectral_conv(on_grid(a_test, N), w), on_grid(op(a_test), N))
        rough = rel_err(spectral_conv(on_grid(a_rough, N), w), on_grid(op(a_rough), N))
        msg.append(f"N={N:3d}: 매끄러운 입력 {fmt(smooth)}, 고주파 입력 {rough:.1e}")
    print(f"{name} | " + " | ".join(msg))

# %% cell 5 — PyTorch FNO: Allen–Cahn 방정식의 한 step 연산자, 해상도와 rollout
import numpy as np
import torch

torch.manual_seed(0); torch.set_num_threads(1)
gen = np.random.default_rng(1)

def allen_cahn(u, n_sub=10, dt=0.01, nu=5e-4):          # u_t = ν u_xx + u - u³, 주기 [0, 1), 반암시적 스펙트럼 step
    k = 2 * np.pi * np.fft.rfftfreq(u.shape[-1], d=1.0 / u.shape[-1])
    for _ in range(n_sub):                              # 한 연산자 step = 시간 0.1
        u = np.fft.irfft((np.fft.rfft(u) + dt * np.fft.rfft(u - u**3)) / (1 + dt * nu * k**2), n=u.shape[-1])
    return u

def trajectories(n, N, steps):                          # 무작위 매끄러운 초기 조건에서 궤적, (n, steps+1, N)
    x, kk = np.arange(N) / N, np.arange(1, 6)
    c = gen.normal(size=(n, 5)) / kk; s = gen.normal(size=(n, 5)) / kk
    u = [0.5 * (c @ np.cos(2 * np.pi * np.outer(kk, x)) + s @ np.sin(2 * np.pi * np.outer(kk, x)))]
    for _ in range(steps):
        u.append(allen_cahn(u[-1]))
    return torch.tensor(np.stack(u, 1), dtype=torch.float32)

class SpectralConv(torch.nn.Module):                    # 낮은 modes개 Fourier 모드에 채널 혼합 가중치
    def __init__(self, ch, modes):
        super().__init__(); self.modes = modes
        self.w = torch.nn.Parameter(torch.randn(ch, ch, modes, dtype=torch.cfloat) / ch)
    def forward(self, v):                               # v: (B, C, N)
        v_hat = torch.fft.rfft(v, norm="forward")[..., :self.modes]          # (B, C, modes)
        out = torch.zeros(v.shape[0], v.shape[1], v.shape[-1] // 2 + 1, dtype=torch.cfloat)
        out[..., :self.modes] = torch.einsum("bik,iok->bok", v_hat, self.w)
        return torch.fft.irfft(out, n=v.shape[-1], norm="forward")

class FNO(torch.nn.Module):
    def __init__(self, ch=24, modes=12):
        super().__init__()
        self.lift, self.proj = torch.nn.Conv1d(1, ch, 1), torch.nn.Conv1d(ch, 1, 1)
        self.spec = torch.nn.ModuleList([SpectralConv(ch, modes) for _ in range(3)])
        self.pw = torch.nn.ModuleList([torch.nn.Conv1d(ch, ch, 1) for _ in range(3)])
    def forward(self, u):                               # u: (B, N) → 다음 step u: (B, N)
        v = self.lift(u[:, None])
        for s, p in zip(self.spec, self.pw):
            v = torch.nn.functional.gelu(s(v) + p(v))
        return u + self.proj(v)[:, 0]                   # 변화량을 예측

train = trajectories(200, 64, 10)                       # 학습: 해상도 64, (200, 11, 64)
X, Y = train[:, :-1].reshape(-1, 64), train[:, 1:].reshape(-1, 64)
model = FNO(); opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for it in range(1500):
    idx = torch.randint(0, X.shape[0], (100,))
    opt.zero_grad(); loss = ((model(X[idx]) - Y[idx]) ** 2).mean(); loss.backward(); opt.step()

rel = lambda a, b: (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()
with torch.no_grad():
    for N in [64, 128]:
        test = trajectories(50, N, 40)                  # 학습하지 않은 초기 조건, 40 step = 시간 4
        u, errs = test[:, 0], []
        for n in range(1, 41):
            u = model(u)
            if n in (10, 20, 40): errs.append(f"{n} step {rel(u, test[:, n]):.1e}")
        print(f"해상도 {N:3d} | 한 step 오차 {rel(model(test[:, 0]), test[:, 1]):.1e} | rollout: " + ", ".join(errs))

# %% cell 6 — 한 step 연산자의 장기 rollout과 제약
dt_shift = 0.01                                         # 한 step = 이류 0.01 (모든 모드의 크기 보존)
u_in = on_grid(a_train, 64)
u_out = on_grid(advect(a_train, dt_shift), 64) + rng.normal(0, 2e-2, u_in.shape)   # 잡음 섞인 학습 쌍
w = fit_spectral(u_in, u_out, K=K_IN)
w_unit = w / np.abs(w)                                  # 제약: |w_k| = 1 (L2 노름 보존)
print(f"학습된 |w_k|의 범위: {np.abs(w).min():.4f} ~ {np.abs(w).max():.4f}  (참값 1)")
u0 = on_grid(a_test[:10], 64)
for steps in [10, 100, 1000]:
    out = []
    for weights in [w, w_unit]:
        u = u0.copy()
        for _ in range(steps):
            u = spectral_conv(u, weights)
        true = on_grid(advect(a_test[:10], dt_shift * steps), 64)
        out.append(f"오차 {rel_err(u, true):.1e}, 노름 비 {np.linalg.norm(u) / np.linalg.norm(u0):.2f}")
    print(f"{steps:4d} step | 제약 없음: {out[0]} | |w|=1 제약: {out[1]}")
