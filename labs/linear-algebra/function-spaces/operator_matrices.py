"""7.9 실습: 함수를 격자 값과 Fourier basis로 각각 표현하고, 미분 연산자를 두 표현에서 행렬로 근사해 비교한다.

실행: uv run --project labs python labs/linear-algebra/function-spaces/operator_matrices.py

1. 격자 표현: 주기 격자 x_i = 2πi/N 위의 값 벡터 u (N,).
   - 유한차분 행렬 D_fd (N, N): (u_{i+1} - u_{i-1}) / (2h)
   - 스펙트럴 미분 행렬 D_sp (N, N) = F^{-1} diag(ik) F (격자 값 → Fourier 계수 → ik 곱 → 격자 값)
2. Fourier 표현: 실수 basis {1, cos x, sin x, ..., cos Kx, sin Kx}의 계수 c (2K+1,).
   미분은 계수 공간에서 블록 대각 행렬 D_c: (a_k, b_k) → (k b_k, -k a_k).
3. 해상도 바꾸기: N = 32 격자에서 '학습한' 3-점 stencil을 N = 64 격자에 그대로 쓰면 어떻게 되는가.
"""
import numpy as np


def grid(N):
    return 2 * np.pi * np.arange(N) / N


def fd_matrix(N):
    h = 2 * np.pi / N
    I = np.eye(N)
    return (np.roll(I, 1, axis=1) - np.roll(I, -1, axis=1)) / (2 * h)


def spectral_matrix(N):
    """격자 값에 작용하는 스펙트럴 미분 행렬 (N, N). 열 j = 단위벡터 e_j의 스펙트럴 미분."""
    k = np.fft.fftfreq(N, d=1.0 / N)
    if N % 2 == 0:
        k[N // 2] = 0.0
    E = np.eye(N)
    return np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(E, axis=0), axis=0))


def fourier_coeffs(u, K, Q=256):
    """u의 실수 Fourier 계수 c = (a_0, a_1, b_1, ..., a_K, b_K), L2 사영(구적점 Q개)."""
    x = grid(Q)
    w = 2 * np.pi / Q
    c = [np.sum(u(x)) * w / (2 * np.pi)]
    for k in range(1, K + 1):
        c += [np.sum(u(x) * np.cos(k * x)) * w / np.pi, np.sum(u(x) * np.sin(k * x)) * w / np.pi]
    return np.array(c)


def eval_fourier(c, x):
    K = (len(c) - 1) // 2
    out = np.full_like(x, c[0])
    for k in range(1, K + 1):
        out += c[2 * k - 1] * np.cos(k * x) + c[2 * k] * np.sin(k * x)
    return out


def coeff_derivative_matrix(K):
    """계수 공간의 미분 행렬 (2K+1, 2K+1): d/dx (a cos kx + b sin kx) = k b cos kx - k a sin kx."""
    D = np.zeros((2 * K + 1, 2 * K + 1))
    for k in range(1, K + 1):
        ia, ib = 2 * k - 1, 2 * k
        D[ia, ib] = k
        D[ib, ia] = -k
    return D


def main():
    u = lambda x: np.exp(np.sin(x))
    du = lambda x: np.cos(x) * np.exp(np.sin(x))

    print("1) 격자 표현: 최대 오차")
    print("     N   유한차분 D_fd   스펙트럴 D_sp")
    for N in [8, 16, 32, 64, 128]:
        x = grid(N)
        e_fd = np.abs(fd_matrix(N) @ u(x) - du(x)).max()
        e_sp = np.abs(spectral_matrix(N) @ u(x) - du(x)).max()
        print(f"  {N:4d}   {e_fd:12.2e}   {e_sp:12.2e}")

    print("\n2) Fourier 표현: 계수 K개까지 자른 뒤 계수 공간에서 미분 (오차는 x 200점에서 최대)")
    print("     K   행렬 크기   u 근사 오차   u' 근사 오차")
    xt = np.linspace(0, 2 * np.pi, 200)
    for K in [2, 4, 8, 12]:
        c = fourier_coeffs(u, K)
        dc = coeff_derivative_matrix(K) @ c
        e_u = np.abs(eval_fourier(c, xt) - u(xt)).max()
        e_du = np.abs(eval_fourier(dc, xt) - du(xt)).max()
        print(f"  {K:4d}   {2 * K + 1:3d}x{2 * K + 1:<3d}   {e_u:11.2e}   {e_du:12.2e}")

    print("\n   유한차분 행렬의 고윳값 허수부 sin(kh)/h와 정확한 값 k (N = 32)")
    N = 32
    h = 2 * np.pi / N
    for k in [1, 4, 8, 15]:
        print(f"   k = {k:2d}   sin(kh)/h = {np.sin(k * h) / h:6.3f}   정확한 값 = {k}")

    print("\n3) 해상도 바꾸기: N = 32에서 학습한 3-점 stencil을 N = 64에 그대로 적용")
    rng = np.random.default_rng(0)
    N_tr = 32
    x = grid(N_tr)
    rows, targets = [], []
    for _ in range(200):                                   # 무작위 저주파 함수로 '학습 데이터'를 만든다
        a, b = rng.normal(size=4), rng.normal(size=4)
        f = sum(a[k] * np.cos((k + 1) * x) + b[k] * np.sin((k + 1) * x) for k in range(4))
        df = sum(-(k + 1) * a[k] * np.sin((k + 1) * x) + (k + 1) * b[k] * np.cos((k + 1) * x) for k in range(4))
        rows.append(np.stack([np.roll(f, 1), f, np.roll(f, -1)], axis=1))   # (u_{i-1}, u_i, u_{i+1})
        targets.append(df)
    w, *_ = np.linalg.lstsq(np.concatenate(rows), np.concatenate(targets), rcond=None)
    print(f"   학습한 stencil w = {np.round(w, 3)}   (참고: ±1/(2h) = ±{1 / (2 * (2 * np.pi / N_tr)):.3f})")
    for N in [32, 64, 128]:
        x = grid(N)
        pred = np.stack([np.roll(u(x), 1), u(x), np.roll(u(x), -1)], axis=1) @ w
        rel = np.linalg.norm(pred - du(x)) / np.linalg.norm(du(x))
        rel_sp = np.linalg.norm(spectral_matrix(N) @ u(x) - du(x)) / np.linalg.norm(du(x))
        print(f"   N = {N:3d}   stencil 상대 오차 = {rel:6.3f}   Fourier 승수 ik 상대 오차 = {rel_sp:.1e}")


if __name__ == "__main__":
    main()
