"""9.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/numerical-methods/interpolation-and-quadrature/cells.py
"""
# %% cell 1 — Runge 현상: 등간격 노드와 Chebyshev 노드
import numpy as np
from scipy.interpolate import BarycentricInterpolator, CubicSpline

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

x_test = np.linspace(-1, 1, 2001)                      # (2001,)
for n in [5, 10, 15, 20]:
    x_eq = np.linspace(-1, 1, n + 1)                   # (n+1,) 등간격
    x_ch = np.cos(np.pi * (2 * np.arange(n + 1) + 1) / (2 * n + 2))   # (n+1,) Chebyshev
    err_eq = np.max(np.abs(BarycentricInterpolator(x_eq, runge(x_eq))(x_test) - runge(x_test)))
    err_ch = np.max(np.abs(BarycentricInterpolator(x_ch, runge(x_ch))(x_test) - runge(x_test)))
    err_sp = np.max(np.abs(CubicSpline(x_eq, runge(x_eq))(x_test) - runge(x_test)))
    print(f"n = {n:2d}  등간격 {err_eq:9.2e}  Chebyshev {err_ch:8.2e}  spline {err_sp:8.2e}")

# %% cell 2 — 보간 안에서는 맞아도 밖에서는 틀린다
rng = np.random.default_rng(0)
x_in = np.linspace(0, 1, 11)                           # (11,) 측정 구간 [0, 1]
y_in = np.sin(2 * np.pi * x_in) + 0.01 * rng.normal(size=11)   # 측정 잡음 0.01
for x_out in [0.5, 0.95, 1.05, 1.1, 1.2]:
    p10 = BarycentricInterpolator(x_in, y_in)(x_out)  # 10차 다항식
    sp = CubicSpline(x_in, y_in)(x_out)
    print(f"x = {x_out:4.2f}  참값 {np.sin(2 * np.pi * x_out):7.3f}  10차 다항식 {p10:7.3f}  spline {sp:7.3f}")

# %% cell 3 — 보간이 아니라 근사: 잡음 있는 데이터
rng = np.random.default_rng(1)
x = np.linspace(-1, 1, 21)                             # (21,)
y = np.exp(x) + 0.05 * rng.normal(size=x.size)         # 잡음 표준편차 0.05
x_fine = np.linspace(-1, 1, 1001)                      # (1001,)
for deg in [2, 4, 8, 20]:
    p = np.polynomial.Chebyshev.fit(x, y, deg)
    rms_fit = np.sqrt(np.mean((p(x) - y) ** 2))
    err = np.max(np.abs(p(x_fine) - np.exp(x_fine)))
    print(f"차수 {deg:2d}  데이터 잔차 RMS {rms_fit:.4f}  참함수와 최대 오차 {err:.4f}")
V_mono = np.vander(x, 21, increasing=True)             # (21, 21) 단항식 기저
V_cheb = np.polynomial.chebyshev.chebvander(x, 20)     # (21, 21) Chebyshev 기저
print(f"cond: 단항식 {np.linalg.cond(V_mono):.1e}  Chebyshev {np.linalg.cond(V_cheb):.1e}")

# %% cell 4 — 수치적분의 수렴 차수
from scipy.integrate import simpson, trapezoid

f, exact = np.exp, np.e - 1.0                          # ∫_0^1 e^x dx
print(" n   사다리꼴    Simpson    Gauss(n점)")
for n in [4, 8, 16]:
    x = np.linspace(0, 1, n + 1)                       # (n+1,) 등간격 n구간
    xg, wg = np.polynomial.legendre.leggauss(n)        # [-1, 1]의 노드·가중치
    gauss = 0.5 * np.sum(wg * f(0.5 * (xg + 1)))       # [0, 1]로 옮김
    e_g = abs(gauss - exact)
    e_g = f"{e_g:.2e}" if e_g > 1e-13 else "< 1e-13"   # 반올림 한계 아래는 판정만
    print(f"{n:2d}  {abs(trapezoid(f(x), x=x) - exact):.2e}  "
          f"{abs(simpson(f(x), x=x) - exact):.2e}  {e_g}")

# %% cell 5 — 차원의 저주: 같은 점 4096개로 [0,1]^d 적분
def exact_cos_sum(d):                                  # ∫ cos(x_1+…+x_d) dx
    return (((np.exp(1j) - 1) / 1j) ** d).real

N = 4096
rng = np.random.default_rng(0)
for d in [1, 2, 3, 4, 6, 12]:
    m = round(N ** (1 / d))                            # 축마다 m점 (m^d = N)
    g = (np.arange(m) + 0.5) / m                       # 중점 규칙 노드
    grid = np.stack(np.meshgrid(*[g] * d), axis=-1).reshape(-1, d)   # (N, d)
    ex = exact_cos_sum(d)
    err_mid = abs(np.mean(np.cos(grid.sum(axis=1))) - ex)
    mc = [np.mean(np.cos(rng.random((N, d)).sum(axis=1))) for _ in range(20)]
    err_mc = np.sqrt(np.mean((np.array(mc) - ex) ** 2))   # 20회 반복의 RMS 오차
    print(f"d = {d:2d}  축당 {m:4d}점  중점격자 오차 {err_mid:.1e}  MC 오차(RMS) {err_mc:.1e}")
