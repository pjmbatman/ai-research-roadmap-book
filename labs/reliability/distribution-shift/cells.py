"""21.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/reliability/distribution-shift/cells.py
"""
# %% cell 1 — 공변량 이동과 중요도 가중
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
f = lambda x: x + 0.3 * x ** 2                       # 참 관계: 조금 휘어 있다
x_s = rng.normal(0.0, 1.0, 500)                      # 학습(source) 입력
y_s = f(x_s) + rng.normal(0, 0.3, 500)
x_t = rng.normal(1.5, 0.6, 20000)                    # 운영(target) 입력: 오른쪽으로 이동
y_t = f(x_t) + rng.normal(0, 0.3, 20000)             # p(y|x)는 그대로 (공변량 이동)

def wls(x, y, w):                                    # 가중 최소제곱 직선 (일부러 틀린 모델)
    A = np.stack([np.ones_like(x), x], axis=1)       # (N, 2)
    return np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))

w_true = norm.pdf(x_s, 1.5, 0.6) / norm.pdf(x_s, 0.0, 1.0)   # 가능도 비 p_t(x) / p_s(x)
ess = w_true.sum() ** 2 / np.sum(w_true ** 2)
for name, w in (("가중 없음", np.ones_like(x_s)), ("중요도 가중", w_true)):
    c = wls(x_s, y_s, w)
    mse_s = np.mean((c[0] + c[1] * x_s - y_s) ** 2)
    mse_t = np.mean((c[0] + c[1] * x_t - y_t) ** 2)
    print(f"{name:7s}  기울기 {c[1]:.2f}  source MSE {mse_s:.3f}  target MSE {mse_t:.3f}")
print(f"유효 표본 수 {ess:.0f} / {len(x_s)}")

# %% cell 2 — 도메인 분류기로 가능도 비 추정
def logreg_newton(F, t, iters=25):                   # F: (N, D) 특징, t: (N,) 0/1 레이블
    b = np.zeros(F.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-F @ b))
        H = F.T @ (F * (p * (1 - p))[:, None]) + 1e-6 * np.eye(F.shape[1])
        b += np.linalg.solve(H, F.T @ (t - p))       # Newton 갱신
    return b

feats = lambda x: np.stack([np.ones_like(x), x, x ** 2], axis=1)   # (N, 3)
x_unlab = x_t[:500]                                  # 레이블 없는 운영 입력 500개면 충분하다
F = feats(np.concatenate([x_s, x_unlab]))
n_s, n_t = len(x_s), len(x_unlab)
t = np.concatenate([np.zeros(n_s), np.ones(n_t)])    # 0: source, 1: target
b = logreg_newton(F, t)
w_hat = np.exp(feats(x_s) @ b) * n_s / n_t           # p(t|x)/p(s|x) × n_s/n_t

c = wls(x_s, y_s, w_hat)
print(f"log 가중치 상관 (추정 vs 참) {np.corrcoef(np.log(w_hat), np.log(w_true))[0, 1]:.3f}")
print(f"추정 가중치로 학습: 기울기 {c[1]:.2f}  target MSE {np.mean((c[0] + c[1] * x_t - y_t) ** 2):.3f}")

# %% cell 3 — 2-표본 검정: 특징별 KS와 분류기 기반 검정
from scipy.stats import ks_2samp, mannwhitneyu

G = lambda X: np.column_stack([np.ones(len(X)), X, X[:, 0] * X[:, 1]])   # 상호작용 특징 포함
L = np.array([[1.0, 0.0], [0.8, 0.6]])               # 상관 0.8을 만드는 Cholesky 인자

def trial(n, shifted):
    A = rng.normal(size=(n, 2))                      # (n, 2) 기준 데이터: 두 특징 독립
    B = rng.normal(size=(n, 2)) @ (L.T if shifted else np.eye(2))  # 각 특징은 여전히 N(0, 1)
    ks = min(ks_2samp(A[:, j], B[:, j]).pvalue for j in range(2)) * 2      # Bonferroni
    h = n // 2                                       # 절반으로 분류기 학습, 나머지 절반으로 검정
    b = logreg_newton(G(np.vstack([A[:h], B[:h]])), np.repeat([0.0, 1.0], h))
    u = mannwhitneyu(G(B[h:]) @ b, G(A[h:]) @ b, alternative="greater")
    return ks < 0.05, u.pvalue < 0.05, u.statistic / (n - h) ** 2

for shifted in (False, True):
    r = np.array([trial(400, shifted) for _ in range(200)])   # (200, 3)
    label = "상관 0 → 0.8" if shifted else "이동 없음  "
    print(f"{label}  기각률: 특징별 KS {r[:, 0].mean():.2f}  분류기 검정 {r[:, 1].mean():.2f}"
          f"   held-out AUC 평균 {r[:, 2].mean():.3f}")

# %% cell 4 — 운영 중 drift 탐지: 배치마다 기준 데이터와 비교
K, START = 100, 60                                   # 배치 100개, 60번째부터 센서가 서서히 어긋난다
ref_temp_c = rng.normal(25.0, 2.0, 2000)             # 배포 시점의 센서 온도 기준 분포
offset_c = np.where(np.arange(K) < START, 0.0, 0.03 * (np.arange(K) - START + 1))
p_batch = np.array([ks_2samp(ref_temp_c, rng.normal(25.0 + o, 2.0, 200)).pvalue for o in offset_c])

for level, name in ((0.05, "α = 0.05"), (0.05 / K, "α = 0.05/100")):
    alarm = p_batch < level
    first = START + int(np.argmax(alarm[START:]))
    print(f"{name:12s}  drift 전 오경보 {alarm[:START].sum()}회   drift 후 첫 경보: 배치 {first}"
          f" (어긋남 {offset_c[first]:.2f} °C)")
