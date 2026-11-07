"""7.9 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linear-algebra/function-spaces/cells.py
"""
# %% cell 1 — 격자 값 벡터의 내적과 L2 내적
import numpy as np

f = lambda x: np.exp(x)
g = lambda x: np.sin(np.pi * x)
exact = np.pi * (np.e + 1) / (np.pi**2 + 1)        # ∫_0^1 e^x sin(πx) dx

for N in [10, 100, 1000]:
    x = (np.arange(N) + 0.5) / N                    # [0, 1]을 N칸으로 나눈 중점, (N,)
    h = 1.0 / N                                     # 칸 너비
    print(f"N = {N:4d}  f·g = {f(x) @ g(x):9.3f}   h (f·g) = {h * (f(x) @ g(x)):.6f}")
print(f"정확한 <f, g> = {exact:.6f}")

# %% cell 2 — basis의 Gram 행렬: 단항식과 Legendre 다항식
xq, wq = np.polynomial.legendre.leggauss(40)        # [-1, 1] 위 Gauss–Legendre 구적점·가중치, (40,)

def gram(B):                                        # B (n, Q): 구적점에서 basis 함수 값
    return (B * wq) @ B.T                           # G_ij = <b_i, b_j>, (n, n)

for n in [4, 8, 12]:
    mono = np.vander(xq, n, increasing=True).T      # 1, x, ..., x^(n-1)   (n, 40)
    leg = np.polynomial.legendre.legvander(xq, n - 1).T   # P_0, ..., P_(n-1)   (n, 40)
    print(f"n = {n:2d}  cond(단항식 Gram) = {np.linalg.cond(gram(mono)):9.2e}"
          f"   cond(Legendre Gram) = {np.linalg.cond(gram(leg)):5.1f}")

G = gram(np.polynomial.legendre.legvander(xq, 3).T)
print("Legendre Gram 대각:", np.round(np.diag(G), 4), " 비대각 ≈ 0:", np.abs(G - np.diag(np.diag(G))).max() < 1e-12)

# %% cell 3 — 미분 연산자의 두 행렬: 유한차분과 Fourier
u = lambda x: np.exp(np.sin(x))                     # 주기 2π인 매끄러운 함수
du = lambda x: np.cos(x) * np.exp(np.sin(x))        # 정확한 도함수

for N in [16, 32, 64]:
    x = 2 * np.pi * np.arange(N) / N                # (N,)
    h = 2 * np.pi / N
    D_fd = (np.roll(np.eye(N), 1, axis=1) - np.roll(np.eye(N), -1, axis=1)) / (2 * h)  # (N, N) 중심 차분
    k = np.fft.fftfreq(N, d=1.0 / N)                # 파수 0, 1, ..., -1
    k[N // 2] = 0.0                                 # 짝수 N의 Nyquist 모드는 0으로
    d_spec = np.real(np.fft.ifft(1j * k * np.fft.fft(u(x))))   # Fourier 계수에 ik를 곱함
    err_fd = np.abs(D_fd @ u(x) - du(x)).max()
    err_sp = np.abs(d_spec - du(x)).max()
    sp = "< 1e-12" if err_sp < 1e-12 else f"{err_sp:.1e}"
    print(f"N = {N:2d}  유한차분 오차 = {err_fd:.2e}   Fourier 오차 = {sp}")

# %% cell 4 — 이산 Laplacian의 고윳값: 낮은 모드와 높은 모드
def neg_laplacian(N):                               # 주기 격자 위 -d²/dx²의 중심 차분 행렬, (N, N)
    h = 2 * np.pi / N
    I = np.eye(N)
    return (2 * I - np.roll(I, 1, axis=1) - np.roll(I, -1, axis=1)) / h**2

lam = np.linalg.eigvalsh(neg_laplacian(64))         # 오름차순, (64,)
for kk in [1, 4, 16, 32]:
    print(f"k = {kk:2d}  연속 고윳값 k² = {kk**2:5d}   이산 고윳값 = {lam[2 * kk - 1]:8.2f}")
for N in [32, 64, 128]:
    print(f"N = {N:3d}  최대 고윳값 = {np.linalg.eigvalsh(neg_laplacian(N)).max():8.1f}")

# %% cell 5 — RBF 커널의 Gram 행렬과 kernel ridge
rng = np.random.default_rng(0)
x_tr = np.sort(rng.uniform(0, 1, 20))               # (N,)
y_tr = np.sin(2 * np.pi * x_tr) + 0.1 * rng.normal(size=20)

def k_rbf(a, b, ell=0.1):                           # (n,), (m,) → (n, m)
    return np.exp(-(a[:, None] - b[None, :]) ** 2 / (2 * ell**2))

K = k_rbf(x_tr, x_tr)                               # (N, N)
print("Gram 최소 고윳값 ≥ 0 :", np.linalg.eigvalsh(K).min() > -1e-10)
alpha = np.linalg.solve(K + 1e-2 * np.eye(20), y_tr)    # (N,) 표현자 계수
x_te = np.linspace(0, 1, 200)
f_te = k_rbf(x_te, x_tr) @ alpha                    # f(x) = Σ_i α_i k(x, x_i), (200,)
print("계수 수 =", alpha.size, "  test RMSE =", round(np.sqrt(np.mean((f_te - np.sin(2 * np.pi * x_te)) ** 2)), 3))
