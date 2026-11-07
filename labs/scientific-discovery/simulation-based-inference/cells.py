"""25.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/scientific-discovery/simulation-based-inference/cells.py
"""
# %% cell 1 — AR(1) 시뮬레이터와 참조 테이블
import numpy as np

T = 50                                            # 시계열 길이
def simulate(rho, rng):                           # rho: (N,) -> x: (N, T)
    x = np.zeros((len(rho), T))
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + rng.normal(size=len(rho))
    return x

def summaries(x):                                 # (N, T) -> (N, 3): 평균, 분산, lag-1 자기상관
    xc = x - x.mean(1, keepdims=True)
    ac1 = (xc[:, 1:] * xc[:, :-1]).sum(1) / (xc**2).sum(1)
    return np.stack([x.mean(1), x.var(1), ac1], axis=1)

rng = np.random.default_rng(1)
rho_true = 0.6
x_obs = simulate(np.array([rho_true]), rng)       # (1, T) 관측 데이터
N = 20000
rho_sim = rng.uniform(0, 1, N)                    # 사전분포 U(0, 1)에서 뽑은 파라미터
x_sim = simulate(rho_sim, rng)                    # (N, T) 참조 테이블: 시뮬레이션 2만 번
S_sim = summaries(x_sim)                          # (N, 3)
s_obs = summaries(x_obs)[0]                       # (3,)

# 비교 기준: 이 장난감 모델은 우도를 계산할 수 있으므로 격자로 정확한 사후분포를 구한다
grid = np.linspace(0, 1, 2001)
loglik = -0.5 * ((x_obs[0, 1:] - grid[:, None] * x_obs[0, :-1]) ** 2).sum(1)
post = np.exp(loglik - loglik.max()); post /= post.sum()
m_exact = (grid * post).sum(); sd_exact = np.sqrt(((grid - m_exact) ** 2 * post).sum())
print(f"관측 요약 통계 평균 {s_obs[0]:.3f}  분산 {s_obs[1]:.3f}  자기상관 {s_obs[2]:.3f}")
print(f"정확한 사후분포  평균 {m_exact:.3f}  표준편차 {sd_exact:.3f}")
# %% cell 2 — ABC rejection: 요약 통계와 허용 오차
def abc(S_sim, s_obs, cols, q=0.01):
    """요약 통계 열 cols로 거리를 재고, 가장 가까운 비율 q의 시뮬레이션을 받아들인다."""
    scale = S_sim[:, cols].std(0)                 # 통계마다 척도가 달라 표준화한다
    d = np.sqrt((((S_sim[:, cols] - s_obs[cols]) / scale) ** 2).sum(1))   # (N,)
    return d <= np.quantile(d, q)                 # (N,) 수용 여부

for name, cols in [("평균", [0]), ("분산", [1]), ("자기상관", [2]), ("셋 다", [0, 1, 2])]:
    acc = abc(S_sim, s_obs, cols)
    r = rho_sim[acc]
    print(f"{name:6s} 수용 {acc.sum()}개  사후 평균 {r.mean():.3f}  표준편차 {r.std():.3f}")
for q in [0.001, 0.01, 0.1, 0.5]:
    r = rho_sim[abc(S_sim, s_obs, [2], q)]
    print(f"자기상관, 수용 비율 {q:5.3f}: 사후 평균 {r.mean():.3f}  표준편차 {r.std():.3f}")
# %% cell 3 — coverage와 구간 폭
def coverage(cols, q, M=300, level=0.9, seed=1):
    rng = np.random.default_rng(seed)
    rho_test = rng.uniform(0, 1, M)                # 사전분포에서 뽑은 '참값'
    S_test = summaries(simulate(rho_test, rng))   # (M, 3)
    hit, width = 0, 0.0
    for m in range(M):                            # 참조 테이블은 다시 쓴다(시뮬레이션 추가 없음)
        r = rho_sim[abc(S_sim, S_test[m], cols, q)]
        lo, hi = np.quantile(r, [(1 - level) / 2, (1 + level) / 2])
        hit += lo <= rho_test[m] <= hi
        width += hi - lo
    return hit / M, width / M

for cols, q in [([2], 0.01), ([2], 0.2), ([2], 0.8)]:
    cov, w = coverage(cols, q)
    print(f"통계 {cols}, q = {q:4.2f}: 90% 구간 coverage {cov:.2f}  평균 폭 {w:.3f}")
# %% cell 4 — 신경 사후 추정(구간 히스토그램)
import warnings
from sklearn.neural_network import MLPClassifier
warnings.simplefilter("ignore")                       # 수렴 경고는 숨긴다

def features(x, lags=5):                              # (N, T) -> (N, lags + 2): 정보를 덜 버리는 넉넉한 특징
    xc = x - x.mean(1, keepdims=True)
    ac = [(xc[:, k:] * xc[:, :-k]).sum(1) / (xc**2).sum(1) for k in range(1, lags + 1)]
    return np.stack(ac + [x.mean(1), np.log(x.var(1))], axis=1)

K = 20                                                # 사후분포를 [0, 1]의 구간 20개 위 히스토그램으로 표현
net = MLPClassifier(hidden_layer_sizes=(32,), max_iter=30, random_state=0)
net.fit(features(x_sim), np.minimum((rho_sim * K).astype(int), K - 1))   # 입력 x -> rho가 속한 구간

def npe_interval(x, level=0.9):                       # (M, T) -> 구간 하한·상한 (M,), (M,)
    cdf = np.cumsum(net.predict_proba(features(x)), axis=1)   # (M, K)
    edges = np.arange(1, K + 1) / K                   # 각 구간의 오른쪽 끝
    lo = np.array([np.interp((1 - level) / 2, np.r_[0, c], np.r_[0, edges]) for c in cdf])
    hi = np.array([np.interp((1 + level) / 2, np.r_[0, c], np.r_[0, edges]) for c in cdf])
    return lo, hi

lo, hi = npe_interval(x_obs)
cdf_exact = np.cumsum(post)                           # 셀 1의 격자 사후분포
print(f"NPE 90% 구간   [{lo[0]:.2f}, {hi[0]:.2f}]")
print(f"정확한 90% 구간 [{np.interp(0.05, cdf_exact, grid):.2f}, {np.interp(0.95, cdf_exact, grid):.2f}]")
rng_t = np.random.default_rng(1)
rho_test = rng_t.uniform(0, 1, 300)
lo, hi = npe_interval(simulate(rho_test, rng_t))      # 새 데이터 300개에 재학습 없이 적용
print(f"NPE 90% 구간 coverage {np.mean((lo <= rho_test) & (rho_test <= hi)):.2f}  평균 폭 {np.mean(hi - lo):.2f}")
# %% cell 5 — 시뮬레이터 오차와 사후 예측 점검
rng = np.random.default_rng(2)
rho_test = rng.uniform(0, 1, 300)
x_clean = simulate(rho_test, rng)                     # (300, T) 시뮬레이터와 같은 과정
x_noisy = x_clean + rng.normal(size=x_clean.shape)    # 시뮬레이터에 없는 측정 잡음이 더해진 장비

for name, x in [("시뮬레이터와 같은 데이터", x_clean), ("측정 잡음이 섞인 데이터", x_noisy)]:
    S = summaries(x)
    hit, p = 0, []
    for m in range(len(S)):
        acc = abc(S_sim, S[m], [2])                   # 자기상관으로 추론
        lo, hi = np.quantile(rho_sim[acc], [0.05, 0.95])
        hit += lo <= rho_test[m] <= hi
        p.append(np.mean(S_sim[acc, 1] >= S[m, 1]))   # 추론에 쓰지 않은 분산으로 사후 예측 점검
    print(f"{name}: 90% coverage {hit / len(S):.2f}   분산 점검 p < 0.05 비율 {np.mean(np.array(p) < 0.05):.2f}")
