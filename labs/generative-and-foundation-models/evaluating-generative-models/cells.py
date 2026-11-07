"""19.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/generative-and-foundation-models/evaluating-generative-models/cells.py
"""
# %% cell 1 — 품질·커버리지·기억을 따로 잰다
import numpy as np

rng = np.random.default_rng(0)
modes = np.array([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])   # 2차원 혼합 가우시안의 중심 (4, 2)
def real(n, std=0.4, use=4):
    return modes[rng.integers(use, size=n)] + std * rng.normal(size=(n, 2))

train, test = real(500), real(500)                 # 생성모델의 학습 자료, 평가용 실제 자료 (500, 2)
models = {"좋은 모델": real(500), "모드 하나 누락": real(500, use=3),
          "흐릿함(std 2배)": real(500, std=0.8), "학습 자료 복사": train + 0.02 * rng.normal(size=(500, 2))}

def dists(a, b):                                   # 쌍별 유클리드 거리 (len(a), len(b))
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))

def precision_recall(real_x, fake_x, k=3):         # k-NN 공 합집합으로 만든 다양체 근사
    r_real = np.sort(dists(real_x, real_x), 1)[:, k]   # 각 실제 점의 k번째 이웃 거리 (자기 자신 제외)
    r_fake = np.sort(dists(fake_x, fake_x), 1)[:, k]
    prec = (dists(fake_x, real_x) <= r_real[None, :]).any(1).mean()   # 가짜가 실제 다양체 안에 있는가
    rec = (dists(real_x, fake_x) <= r_fake[None, :]).any(1).mean()    # 실제가 가짜 다양체 안에 있는가
    return prec, rec

base_nn = np.median(dists(test, train).min(1))     # 실제 새 표본이 학습 자료에서 떨어진 거리
print("모델               precision  recall   학습 자료까지 거리 / 기준")
for name, fake in models.items():
    p, r = precision_recall(test, fake)
    print(f"{name:14}     {p:.3f}     {r:.3f}    {np.median(dists(fake, train).min(1)) / base_nn:.2f}")

# %% cell 2 — likelihood는 표본 품질과 따로 움직인다
from scipy.stats import multivariate_normal

d = 64
x_te = rng.normal(size=(2000, d))                  # 실제 자료: N(0, I_64)
logp_good = multivariate_normal(np.zeros(d), np.eye(d)).logpdf(x_te)
logp_junk = multivariate_normal(np.zeros(d), 25 * np.eye(d)).logpdf(x_te)
for w in [1.0, 0.5, 0.01]:                         # 좋은 모델을 w만큼, 쓰레기(분산 25배)를 1-w만큼 섞는다
    mix = np.logaddexp(np.log(w) + logp_good, np.log1p(-w) + logp_junk) if w < 1 else logp_good
    print(f"좋은 표본 비율 {w:4.2f}   평균 log-likelihood/차원 {mix.mean() / d:.4f} nat")
print(f"상한 차이 log(100)/d = {np.log(100) / d:.4f} nat")

# %% cell 3 — MMD 순열 검정: 대역폭과 표본 수가 검정력을 정한다
def mmd2_with_perms(x, y, bw=None, n_perm=100):    # 불편 MMD²와 순열 null 분포, RBF kernel
    z, n = np.vstack([x, y]), len(x)
    D2 = dists(z, z) ** 2                          # (2n, 2n)
    bw = np.sqrt(np.median(D2[D2 > 0])) if bw is None else bw   # 없으면 거리의 중앙값
    K = np.exp(-D2 / (2 * bw**2))                  # 대각 = 1
    E = np.zeros((2 * n, n_perm + 1))              # 열마다 "X 쪽" 표시: 0열 = 원래 배정
    E[:n, 0] = 1
    for j in range(1, n_perm + 1):
        E[rng.permutation(2 * n)[:n], j] = 1
    F = 1 - E
    kxx, kyy, kxy = (E * (K @ E)).sum(0), (F * (K @ F)).sum(0), (E * (K @ F)).sum(0)
    m = (kxx - n + kyy - n) / (n * (n - 1)) - 2 * kxy / n**2    # 대각(자기 자신) 항을 뺀 불편 추정
    return m[0], m[1:]

def reject_rate(std, bw, n, reps=20):              # 20번 반복한 실험 중 p < 0.05인 비율
    out = []
    for _ in range(reps):
        stat, null = mmd2_with_perms(real(n, std=std), real(n), bw)
        out.append((1 + np.sum(null >= stat)) / (1 + len(null)) < 0.05)
    return np.mean(out)

print("대역폭      n    기각률(같은 분포)   기각률(std 1.5배)")
for bw in [None, 0.5]:
    for n in [30, 200]:
        print(f"{'중앙값' if bw is None else bw:>5}    {n:4d}        {reject_rate(0.4, bw, n):.2f}              {reject_rate(0.6, bw, n):.2f}")

# %% cell 4 — FID의 가우시안 근사: 무엇을 보고 무엇을 놓치는가
from scipy.linalg import sqrtm

def frechet(a, b):                                 # 두 특징 집합에 가우시안을 맞춘 뒤의 Fréchet 거리
    mu_a, mu_b = a.mean(0), b.mean(0)
    Sa, Sb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = sqrtm(Sa @ Sb).real                  # (Sa Sb)^{1/2}
    return np.sum((mu_a - mu_b) ** 2) + np.trace(Sa + Sb - 2 * covmean)

W_f = rng.normal(size=(2, 32))
feat = lambda x: np.maximum(0, x @ W_f + 0.5)      # 작은 무작위 ReLU 특징 (N, 32)
gens = {"좋은 모델": lambda n: real(n), "모드 하나 누락": lambda n: real(n, use=3),
        "흐릿함(std 2배)": lambda n: real(n, std=0.8),
        "학습 자료 복사": lambda n: train[:n] + 0.02 * rng.normal(size=(n, 2)),
        "평균·공분산만 맞춘 가우시안": lambda n: train.mean(0) + rng.normal(size=(n, 2)) @ np.linalg.cholesky(np.cov(train.T)).T}
print("모델 (5회 반복 평균)          원 좌표   ReLU 특징   ReLU 특징·학습 자료 기준")
for name, g in gens.items():
    res = np.array([(frechet(f, r), frechet(feat(f), feat(r)), frechet(feat(f), feat(train)))
                    for f, r in ((g(500), real(500)) for _ in range(5))]).mean(0)
    print(f"{name:22}   {res[0]:6.3f}    {res[1]:6.2f}       {res[2]:6.2f}")
for n in (50, 500):
    same = [frechet(feat(real(n)), feat(real(n))) for _ in range(5)]
    print(f"같은 분포 두 표본 n = {n:3d}: FID(ReLU 특징) {np.mean(same):.2f}")

# %% cell 5 — 생성된 시계열: 통계·물리 일관성·하위 과제
t_s = np.linspace(0, 1, 64)                        # 시간 (T,), 단위 s
def oscillators(n, f_hz=(3, 8)):                   # 감쇠 진동 (n, T)과 각진동수 ω (n,), 단위 rad/s
    om = 2 * np.pi * rng.uniform(*f_hz, n); ga = rng.uniform(0.5, 4.0, n); ph = rng.uniform(0, 2 * np.pi, n)
    return np.exp(-ga[:, None] * t_s) * np.cos(om[:, None] * t_s + ph[:, None]) + 0.05 * rng.normal(size=(n, 64)), om

X_real, om_real = oscillators(1000); X_te, om_te = oscillators(1000)
X_shuf = np.stack([col[rng.permutation(1000)] for col in X_real.T], 1)   # 시점마다 다른 계열에서 값을 가져온다
gens_ts = {"좋은 모델": oscillators(1000), "시점별 섞기": (X_shuf, om_real), "좁은 주파수(3–5 Hz)": oscillators(1000, (3, 5))}
spec = lambda X: np.abs(np.fft.rfft(X, axis=1))   # 진폭 스펙트럼 특징 (n, 33)
def tstr_r2(Xg, yg):                               # 생성 자료로 학습(ridge), 실제 자료로 시험
    F, Ft = spec(Xg), spec(X_te)
    w = np.linalg.solve(F.T @ F + 1e-2 * len(F) * np.eye(F.shape[1]), F.T @ (yg - yg.mean()))
    return 1 - np.mean((om_te - Ft @ w - yg.mean()) ** 2) / np.var(om_te)

lag1 = lambda X: np.mean(np.sum(X[:, 1:] * X[:, :-1], 1) / np.sum(X**2, 1))   # 평균 lag-1 자기상관
def peak_hz(X):                                    # 계열마다 가장 강한 주파수의 5–95% 범위, 단위 Hz
    f = (np.argmax(spec(X)[:, 1:], 1) + 1) * 63 / 64  # 표본 간격 1/63 s, 64점 → bin 간격 63/64 Hz
    return f"{np.percentile(f, 5):4.1f}–{np.percentile(f, 95):4.1f}"
print(f"{'':16} 시점별 평균 오차  lag-1 자기상관  주된 주파수(Hz)  TSTR R²(ω)")
print(f"{'실제 자료':16}       -            {lag1(X_te):.3f}        {peak_hz(X_te)}      {tstr_r2(X_real, om_real):.3f}")
for name, (Xg, yg) in gens_ts.items():
    print(f"{name:16}     {np.abs(Xg.mean(0) - X_te.mean(0)).max():.3f}          {lag1(Xg):.3f}        {peak_hz(Xg)}      {tstr_r2(Xg, yg):.3f}")
