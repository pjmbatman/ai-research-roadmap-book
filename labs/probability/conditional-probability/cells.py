"""11.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/probability/conditional-probability/cells.py
"""
# %% cell 1 — 결합분포 표에서 주변분포와 조건부분포
import numpy as np

# 행: 구름 X (0 맑음, 1 구름 조금, 2 흐림), 열: 비 Y (0 안 옴, 1 옴)
P = np.array([[0.40, 0.01],
              [0.25, 0.04],
              [0.15, 0.15]])                   # (3, 2) 결합분포 p(x, y)
p_x = P.sum(axis=1)                            # (3,) 주변화: y에 대해 합
p_y = P.sum(axis=0)                            # (2,)
p_y_given_x = P / p_x[:, None]                 # (3, 2) 각 행을 p(x)로 나눈다
print("p(x)       =", p_x, "  p(y) =", p_y)
print("p(비 | x)  =", np.round(p_y_given_x[:, 1], 3))
print("곱의 법칙 p(y|x) p(x) == p(x, y):", np.allclose(p_y_given_x * p_x[:, None], P))
print("E[Y] =", p_y[1], "  sum_x E[Y|x] p(x) =", round(float(p_y_given_x[:, 1] @ p_x), 3))

# %% cell 2 — 베이즈 정리: 진단 검사와 기저율
prev, sens, fpr = 0.001, 0.99, 0.05            # 유병률, 민감도 P(+|병), 위양성률 P(+|정상)
post = sens * prev / (sens * prev + fpr * (1 - prev))
print(f"베이즈 정리  P(병 | +) = {post:.4f}")

rng = np.random.default_rng(0)
n = 2_000_000
sick = rng.random(n) < prev                                    # (n,) 실제 상태
positive = np.where(sick, rng.random(n) < sens, rng.random(n) < fpr)   # (n,) 검사 결과
print(f"시뮬레이션  양성 {positive.sum():,}명 중 환자 {(sick & positive).sum():,}명"
      f" → {np.mean(sick[positive]):.4f}")
print(f"P(+ | 병) = {np.mean(positive[sick]):.3f}  vs  P(병 | +) = {np.mean(sick[positive]):.3f}")

# %% cell 3 — 상관 0이지만 독립이 아닌 경우, 조건부 독립
x = rng.uniform(-1, 1, size=200_000)                         # (N,)
y = x**2                                                     # y는 x로 완전히 정해진다
print(f"corr(X, Y) = {np.corrcoef(x, y)[0, 1]:+.3f}")
print(f"P(Y > 0.5) = {np.mean(y > 0.5):.3f}   P(Y > 0.5 | |X| > 0.8) = {np.mean(y[np.abs(x) > 0.8] > 0.5):.3f}")

fault = rng.random(200_000) < 0.1                            # 공통 원인: 장비 고장
a = rng.random(200_000) < np.where(fault, 0.9, 0.05)         # 센서 A 경보
b = rng.random(200_000) < np.where(fault, 0.8, 0.05)         # 센서 B 경보
print(f"전체:       corr(A, B) = {np.corrcoef(a, b)[0, 1]:+.3f}")
for f in (False, True):
    m = fault == f
    print(f"고장 = {f!s:5}: corr(A, B) = {np.corrcoef(a[m], b[m])[0, 1]:+.3f}")

# %% cell 4 — collider: 독립인 두 원인이 결과를 조건으로 하면 의존한다
n = 500_000
leak = rng.random(n) < 0.05                                  # 원인 1: 누설
drift = rng.random(n) < 0.10                                 # 원인 2: 센서 drift (누설과 독립)
alarm = leak | drift                                         # 결과: 압력 경보
print(f"P(누설)               = {leak.mean():.3f}")
print(f"P(누설 | drift)       = {leak[drift].mean():.3f}   (조건 없음: 독립)")
print(f"P(누설 | 경보)         = {leak[alarm].mean():.3f}")
print(f"P(누설 | 경보, drift)  = {leak[alarm & drift].mean():.3f}   (경보를 조건으로: drift가 누설을 '설명해 버린다')")
