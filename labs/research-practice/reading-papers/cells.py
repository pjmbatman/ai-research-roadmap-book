"""23.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-practice/reading-papers/cells.py
"""
# %% cell 1 — 논문 표의 수치로 효과 크기와 신뢰구간 다시 계산
import numpy as np
from scipy import stats

n = 5                                        # 표 캡션: "seed 5개의 평균 ± ..."
m_base, pm_base = 0.842, 0.006               # baseline 정확도
m_ours, pm_ours = 0.854, 0.005               # 제안 방법 정확도

def compare(s_base, s_ours):
    diff = m_ours - m_base
    se = np.sqrt(s_base**2 / n + s_ours**2 / n)            # Welch 표준오차
    v = se**4 / ((s_base**2 / n) ** 2 / (n - 1) + (s_ours**2 / n) ** 2 / (n - 1))
    t = stats.t.ppf(0.975, v)
    d = diff / np.sqrt((s_base**2 + s_ours**2) / 2)        # Cohen's d (pooled SD)
    return diff, diff - t * se, diff + t * se, d

for label, sb, so in [("±가 표준편차(SD)", pm_base, pm_ours),
                      ("±가 표준오차(SE)", pm_base * np.sqrt(n), pm_ours * np.sqrt(n))]:
    diff, lo, hi, d = compare(sb, so)
    print(f"{label}: 차이 {diff:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], d = {d:.2f}")

# %% cell 2 — 논문 수식과 공개 코드의 정규화 계수
rng = np.random.default_rng(0)
N, D = 200, 10
X = rng.normal(size=(N, D))                  # (N, D)
w_true = rng.normal(size=D)                  # (D,)
y = X @ w_true + rng.normal(size=N)          # (N,)
X_te = rng.normal(size=(2000, D))
y_te = X_te @ w_true + rng.normal(size=2000)

lam = 1.0
# 논문: L = (1/N) sum (y - Xw)^2 + lam |w|^2  ->  (X^T X / N + lam I) w = X^T y / N
w_paper = np.linalg.solve(X.T @ X / N + lam * np.eye(D), X.T @ y / N)
# 공개 코드: loss = ((y - X @ w) ** 2).sum() + lam * (w ** 2).sum()  ->  (X^T X + lam I) w = X^T y
w_code = np.linalg.solve(X.T @ X + lam * np.eye(D), X.T @ y)
w_match = np.linalg.solve(X.T @ X + N * lam * np.eye(D), X.T @ y)   # 코드에서 lam을 N배로

mse = lambda w: np.mean((y_te - X_te @ w) ** 2)
print(f"|w| 참값 {np.linalg.norm(w_true):.2f}  논문 식 {np.linalg.norm(w_paper):.2f}  코드 {np.linalg.norm(w_code):.2f}")
print(f"test MSE  논문 식 {mse(w_paper):.3f}  코드 {mse(w_code):.3f}")
print("코드의 lam을 N배로 하면 논문 식과 같다:", np.allclose(w_match, w_paper))
