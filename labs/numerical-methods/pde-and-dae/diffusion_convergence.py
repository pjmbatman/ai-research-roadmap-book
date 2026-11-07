"""9.7 실습: 1차원 확산 방정식을 method of lines로 풀고 격자 수렴 연구를 수행한다.

u_t = α u_xx (0 < x < 1), u(0, t) = u(1, t) = 0
1) 해석해가 있는 경우(u0 = sin πx + 0.5 sin 3πx): 격자를 바꾸며 오차와 관측 차수를 잰다.
2) 시간 tolerance가 느슨하면 격자를 아무리 줄여도 오차가 줄지 않는 것(시간 오차가 지배)을 본다.
3) 해석해가 없는 경우(비선형 확산계수): 세 격자의 결과로 관측 차수·Richardson 외삽·GCI를 계산한다.

실행: uv run --project labs python labs/numerical-methods/pde-and-dae/diffusion_convergence.py
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

ALPHA, T_END = 1.0, 0.05


def laplacian(N):
    """균일 격자(구간 N개)의 내부점 N-1개에 대한 2차 중심차분 라플라시안, shape (N-1, N-1)."""
    return diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N - 1, N - 1)) / (1.0 / N) ** 2


def exact(x, t):
    return (np.exp(-ALPHA * np.pi**2 * t) * np.sin(np.pi * x)
            + 0.5 * np.exp(-9 * ALPHA * np.pi**2 * t) * np.sin(3 * np.pi * x))


def solve_linear(N, rtol, atol):
    x = np.linspace(0, 1, N + 1)[1:-1]       # (N-1,)
    L = ALPHA * laplacian(N)
    sol = solve_ivp(lambda t, u: L @ u, (0, T_END), exact(x, 0.0), method="BDF", jac=L, rtol=rtol, atol=atol)
    return np.max(np.abs(sol.y[:, -1] - exact(x, T_END))), sol.t.size - 1


def grid_study(rtol, atol, label):
    print(f"\n[{label}: rtol {rtol:.0e}, atol {atol:.0e}]")
    prev = None
    for N in [10, 20, 40, 80, 160]:
        err, steps = solve_linear(N, rtol, atol)
        order = "" if prev is None else f"관측 차수 {np.log2(prev / err):5.2f}"
        print(f"N = {N:3d}  Δx = {1 / N:.4f}  최대 오차 {err:.3e}  {order:14s}  시간 step {steps}")
        prev = err


def solve_nonlinear(N, T=0.05):
    """u_t = (D(u) u_x)_x, D(u) = 1 + u². 보존형 유한체적식: 면에서의 D는 이웃 두 점 D의 평균."""
    x = np.linspace(0, 1, N + 1)             # (N+1,) 경계 포함
    dx = 1.0 / N

    def rhs(t, u_in):
        u = np.concatenate([[0.0], u_in, [0.0]])      # (N+1,) 경계값 0
        D = 1.0 + u**2
        D_face = 0.5 * (D[1:] + D[:-1])               # (N,) 면 i+1/2
        flux = -D_face * np.diff(u) / dx              # (N,) 확산 플럭스
        return -np.diff(flux) / dx                    # (N-1,) 내부점

    u0 = np.sin(np.pi * x[1:-1]) + 0.5 * np.sin(3 * np.pi * x[1:-1])
    sol = solve_ivp(rhs, (0, T), u0, method="BDF", rtol=1e-10, atol=1e-12)
    u_end = sol.y[:, -1]
    # 관심 출력: 가운데 값과 전체 열량(∫u dx, 사다리꼴)
    return u_end[(N - 1) // 2], dx * np.sum(u_end)


def richardson(f1, f2, f3, r=2.0, Fs=1.25):
    """거친(f1) → 고운(f3) 세 격자 결과에서 관측 차수 p, 외삽값, 고운 격자의 GCI(상대값)를 계산한다."""
    p = np.log(abs(f2 - f1) / abs(f3 - f2)) / np.log(r)
    f_ext = f3 + (f3 - f2) / (r**p - 1)
    gci = Fs * abs((f3 - f2) / f3) / (r**p - 1)
    return p, f_ext, gci


def main():
    grid_study(1e-10, 1e-12, "엄격한 시간 tolerance")
    grid_study(1e-3, 1e-6, "solve_ivp 기본 tolerance")

    print("\n[해석해가 없는 비선형 확산: 격자 40, 80, 160]")
    res = {N: solve_nonlinear(N) for N in [40, 80, 160, 320]}
    for k, name in enumerate(["u(0.5, T)", "∫u dx"]):
        f1, f2, f3 = (res[N][k] for N in (40, 80, 160))
        p, f_ext, gci = richardson(f1, f2, f3)
        print(f"{name:10s} N=40 {f1:.8f}  N=80 {f2:.8f}  N=160 {f3:.8f}")
        print(f"{'':10s} 관측 차수 {p:.3f}  외삽값 {f_ext:.8f}  GCI(N=160) {100 * gci:.4f}%"
              f"  | 확인용 N=320 {res[320][k]:.8f}")


if __name__ == "__main__":
    main()
