"""9.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/numerical-methods/floating-point/cells.py
"""
# %% cell 1 — 형식별 정밀도와 범위, 0.1의 실제 저장값
import numpy as np

for dt in [np.float64, np.float32, np.float16]:
    fi = np.finfo(dt)
    print(f"{fi.dtype.name:8s} 가수 {fi.nmant:2d}비트  eps = {fi.eps:.2e}  "
          f"최대 = {fi.max:.2e}  최소 정규수 = {fi.smallest_normal:.2e}")

print(f"float32(0.1)의 저장값 = {float(np.float32(0.1)):.20f}")
print(f"float64(0.1)의 저장값 = {0.1:.20f}")
print("0.1 + 0.2 == 0.3 :", 0.1 + 0.2 == 0.3)
for v in [1.0, 1024.0, 2.0**24]:
    print(f"x = {v:>10.0f}  float32의 ulp = {np.spacing(np.float32(v)):.3e}")
print("float32(2**24 + 1) =", int(np.float32(2**24 + 1)))

# %% cell 2 — 절단 오차와 반올림 오차: 전진 차분의 h 선택
x0 = 1.0
exact = np.cos(x0)                      # d/dx sin(x)
for h in [1e-1, 1e-4, 1e-8, 1e-12]:
    approx = (np.sin(x0 + h) - np.sin(x0)) / h
    print(f"h = {h:.0e}  상대 오차 = {abs(approx - exact) / abs(exact):.2e}")

# %% cell 3 — 파국적 상쇄와 fp16의 overflow·underflow
for x in [1e-2, 1e-5, 1e-8]:
    naive = (1 - np.cos(x)) / x**2                 # 1에 가까운 두 수의 뺄셈
    stable = 2 * np.sin(x / 2) ** 2 / x**2         # 같은 식, 뺄셈 없음
    print(f"x = {x:.0e}  naive = {naive:.10f}  stable = {stable:.10f}")

grad = 1e-8                              # 작은 gradient 값
print("float16(1e-8)          =", np.float16(grad))           # underflow → 0
print("float16(1e-8 * 2**16)  =", np.float16(grad * 2**16))   # loss scaling
with np.errstate(over="ignore"):
    print("float16(300) ** 2      =", np.float16(300) ** 2)   # overflow → inf

# %% cell 4 — 합의 순서와 Kahan 보정 합
rng = np.random.default_rng(0)
vals = rng.uniform(0, 1, 100_000).astype(np.float32)        # (100000,)
exact = np.sum(vals.astype(np.float64))

def naive_sum(a):
    s = np.float32(0)
    for v in a:
        s += v
    return s

def kahan_sum(a):
    s, c = np.float32(0), np.float32(0)
    for v in a:
        y = v - c              # 지난번에 잃은 부분을 보정
        t = s + y
        c = (t - s) - y        # 이번 덧셈에서 잃은 부분
        s = t
    return s

for name, s in [("순서대로", naive_sum(vals)), ("Kahan", kahan_sum(vals)),
                ("np.sum(쌍별)", np.sum(vals))]:
    print(f"{name:12s} 상대 오차 = {abs(float(s) - exact) / exact:.1e}")

# %% cell 5 — log-sum-exp, log1p, 분산의 안정한 계산
z = np.array([1000.0, 1001.0, 1002.0])            # (3,) logits

with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(z) / np.sum(np.exp(z))
print("naive softmax :", naive)

m = np.max(z)
lse = m + np.log(np.sum(np.exp(z - m)))           # log-sum-exp
print("stable softmax:", np.exp(z - lse))
print("log-sum-exp   :", lse)

x = 1e-12
print(f"log(1 + x) = {np.log(1 + x):.6e}   log1p(x) = {np.log1p(x):.6e}")

t = 1e8 + rng.normal(0, 1, 1000)                  # (1000,) 큰 offset + 분산 1
one_pass = np.mean(t**2) - np.mean(t) ** 2        # E[t²] - E[t]²
two_pass = np.mean((t - np.mean(t)) ** 2)         # 평균을 먼저 뺀다
print(f"분산: 한 번 훑기 = {one_pass:.4f}   두 번 훑기 = {two_pass:.4f}")
