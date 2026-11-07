"""11.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/probability/stochastic-processes/cells.py
"""
# %% cell 1 — 전이행렬과 정상분포
import numpy as np

P = np.array([[0.90, 0.08, 0.02],          # 상태 A, B, C 사이의 한 step 전이확률
              [0.10, 0.80, 0.10],          # 행 i = 현재 상태, 열 j = 다음 상태
              [0.05, 0.15, 0.80]])         # (3, 3), 각 행의 합 1
evals, evecs = np.linalg.eig(P.T)          # 정상분포는 P^T의 고윳값 1 고유벡터
pi = np.real(evecs[:, np.argmax(np.real(evals))])
pi = pi / pi.sum()
print("정상분포 pi        ", np.round(pi, 4), "  pi P = pi:", np.allclose(pi @ P, pi))
print("고윳값 크기         ", np.round(np.sort(np.abs(evals))[::-1], 4))

p = np.array([0.0, 0.0, 1.0])              # C에서 출발
for k in range(1, 41):
    p = p @ P
    if k in (1, 5, 10, 20, 40):
        print(f"k = {k:2d}  분포 {np.round(p, 4)}   |p_k - pi|_1 = {np.abs(p - pi).sum():.2e}")

rng = np.random.default_rng(0)
cum = P.cumsum(axis=1)                     # (3, 3) 행마다 누적확률
state, counts = 2, np.zeros(3)
for u in rng.uniform(size=100_000):        # 한 경로를 길게 시뮬레이션 (역변환 표집)
    state = int(np.sum(u > cum[state]))
    counts[state] += 1
print("시간 평균 점유율    ", np.round(counts / counts.sum(), 4))

# %% cell 2 — Metropolis: 정규화 상수 없이 이중 우물 분포 표집
from scipy import integrate

kT = 0.15
def log_target(x):                         # log p(x) + 상수, U(x) = (x^2 - 1)^2
    return -((x**2 - 1) ** 2) / kT

Z, _ = integrate.quad(lambda x: np.exp(log_target(x)), -4, 4)
ex2, _ = integrate.quad(lambda x: x**2 * np.exp(log_target(x)) / Z, -4, 4)
print(f"수치적분 E[x^2] = {ex2:.4f}")
for step in (0.05, 0.5, 2.0):
    x, chain, acc = 1.0, np.empty(20_000), 0
    for t in range(chain.size):
        prop = x + step * rng.normal()     # 대칭 제안
        if np.log(rng.uniform()) < log_target(prop) - log_target(x):
            x, acc = prop, acc + 1
        chain[t] = x
    print(f"step {step:4.2f}  수용률 {acc / chain.size:.2f}  E[x^2] {np.mean(chain**2):.4f}"
          f"  오른쪽 우물 비율 {np.mean(chain > 0):.3f}")

# %% cell 3 — 포아송 과정과 random walk의 연속시간 극한
rate_per_s, T_s = 3.0, 10.0
gaps_s = rng.exponential(1 / rate_per_s, size=(20_000, 60))      # (경로, 사건) 도착 간격
counts = (np.cumsum(gaps_s, axis=1) <= T_s).sum(axis=1)          # 각 경로에서 [0, T] 사건 수
print(f"포아송 과정 N(10 s): 평균 {counts.mean():.3f}  분산 {counts.var():.3f}  (이론 둘 다 {rate_per_s * T_s})")

n_paths, dt = 20_000, 1e-2
n_steps = int(1.0 / dt)
steps = rng.choice([-1.0, 1.0], size=(n_paths, n_steps)) * np.sqrt(dt)   # +-sqrt(dt) 걸음
B = np.cumsum(steps, axis=1)                                     # (n_paths, n_steps), B(t_k)
i03, i10 = round(0.3 / dt) - 1, n_steps - 1                    # B(0.3), B(1)의 열 번호
print(f"Var B(1)          = {B[:, i10].var():.4f}   (이론 1)")
print(f"Cov(B(0.3), B(1)) = {np.cov(B[:, i03], B[:, i10])[0, 1]:.4f}   (이론 min(0.3, 1) = 0.3)")
dB = np.diff(B[:, 9::10], axis=1)                                # 간격 0.1인 증분
print(f"겹치지 않는 증분의 상관 = {np.corrcoef(dB[:, 2], dB[:, 7])[0, 1]:.4f}   (이론 0)")

# %% cell 4 — Euler–Maruyama로 Ornstein–Uhlenbeck 과정 시뮬레이션
theta, sigma, dt = 2.0, 1.0, 0.01                # dX = -theta X dt + sigma dW
n_paths, n_steps = 5_000, 400                    # t = 0 ~ 4
x = np.full(n_paths, 3.0)                        # (n_paths,) X(0) = 3
for k in range(1, n_steps + 1):
    x = x - theta * x * dt + sigma * np.sqrt(dt) * rng.normal(size=n_paths)
    if k in (25, 50, 100, 400):
        t = k * dt
        print(f"t = {t:4.2f}  평균 {x.mean():6.3f} (이론 {3 * np.exp(-theta * t):6.3f})"
              f"  분산 {x.var():.4f} (이론 {sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * t)):.4f})")

# %% cell 5 — 정상 AR(1)과 random walk: 서로 무관한 두 시계열의 상관
from scipy.signal import lfilter

for phi in (0.5, 1.0):
    e = rng.normal(size=(2, 1_000, 500))                 # (시계열 쌍, 반복, 시간)
    y = lfilter([1.0], [1.0, -phi], e, axis=-1)          # y_t = phi y_{t-1} + e_t
    a = y[0] - y[0].mean(axis=1, keepdims=True)          # (1000, 500)
    b = y[1] - y[1].mean(axis=1, keepdims=True)
    r = np.abs((a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1)))   # (1000,)
    print(f"phi = {phi}:  독립인 두 시계열의 |상관| 중앙값 {np.median(r):.3f}   |상관| > 0.5 비율 {np.mean(r > 0.5):.3f}")
