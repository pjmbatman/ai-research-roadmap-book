"""28.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

합성 분자 공간(비트 벡터)과 합성 목적함수로 생성·최적화·평가를 보인다.

실행: uv run --project labs python labs/chemistry-and-materials/molecular-design/cells.py
"""
# %% cell 1 — 합성 분자 공간과 목적함수, 유효성·합성 가능성
import numpy as np

L = 24                                             # 조각(fragment) 종류 수
rng = np.random.default_rng(0)
w = np.abs(rng.normal(0, 1.0, L)) + 0.5            # 조각마다 양의 물성 기여
crowd = np.abs(rng.normal(0, 0.6, (L, L))); crowd = np.triu(crowd, 1)   # 조각 쌍의 혼잡 penalty(양수)
incompat = {(2, 5), (7, 11), (13, 19)}             # 함께 있으면 안 되는 조각 쌍(유효성 규칙)

def valid(x):                                       # 화학적 타당성: 금지 쌍이 없고 조각 1–8개
    on = np.flatnonzero(x)
    if not (1 <= len(on) <= 8):
        return False
    return not any((i, j) in incompat for i in on for j in on if i < j)

def true_property(x):                               # 참 물성: 조각 기여 − 혼잡 penalty(조각이 몰리면 감소)
    return x @ w - x @ crowd @ x

def sa_score(x):                                    # 합성 가능성: 조각이 많을수록, 쌍이 몰릴수록 어렵다
    return 1.0 + 0.5 * x.sum() + 0.4 * (x @ (crowd > 0).astype(float) @ x)   # 1(쉬움)~10(어려움)

rng2 = np.random.default_rng(1)
sample = (rng2.random((2000, L)) < 0.15).astype(float)
val = np.array([valid(x) for x in sample])
print(f"무작위 비트 벡터 2000개 중 유효한 분자: {val.sum()} ({val.mean():.0%})")
props = np.array([true_property(x) for x in sample[val]])
sas = np.array([sa_score(x) for x in sample[val]])
print(f"유효 분자의 참 물성 범위 [{props.min():.2f}, {props.max():.2f}], SA 점수 범위 [{sas.min():.2f}, {sas.max():.2f}]")

# %% cell 2 — 유전 알고리즘: 합성 가능성 penalty의 유무로 결과가 달라진다
def ga(seed, obj, pop=60, gens=40):
    r = np.random.default_rng(seed)
    P = (r.random((pop, L)) < 0.12).astype(float)
    for _ in range(gens):
        f = np.array([obj(x) for x in P])
        P = P[np.argsort(f)[::-1]]
        parents = P[:pop // 2]                        # 상위 절반이 부모
        kids = []
        for _ in range(pop - len(parents)):
            a, b = parents[r.integers(len(parents))], parents[r.integers(len(parents))]
            child = np.where(r.random(L) < 0.5, a, b).copy()   # 균일 교차
            flip = r.random(L) < 0.06                 # 돌연변이
            child[flip] = 1 - child[flip]
            kids.append(child)
        P = np.vstack([parents, kids])
    return P[0]

prop_only = lambda x: true_property(x) if valid(x) else -1e9
with_sa = lambda x: true_property(x) - 0.8 * sa_score(x) if valid(x) else -1e9
for name, obj in [("물성만 최적화     ", prop_only), ("물성 − 0.8×SA 최적화", with_sa)]:
    winners = [ga(s, obj) for s in range(5)]
    p = np.mean([true_property(x) for x in winners])
    sa = np.mean([sa_score(x) for x in winners])
    print(f"{name}: 평균 물성 {p:+.2f}, 평균 SA 점수 {sa:.2f}")

# %% cell 3 — 예측 모델 착취: 학습 분포 밖으로 최적화하면 모델이 과신한다
from numpy.linalg import lstsq

n_frag = sample[val].sum(1)
lib = sample[val][n_frag <= 3][:400]               # 라이브러리: 조각 3개 이하의 '단순한' 분자만
y_lib = np.array([true_property(x) for x in lib])
w_hat, *_ = lstsq(np.c_[lib, np.ones(len(lib))], y_lib, rcond=None)   # 선형 대리 모델(혼잡은 거의 못 봄)
surrogate = lambda X: np.atleast_2d(X) @ w_hat[:L] + w_hat[L]

def tanimoto_max(X, ref):                           # ref와의 최대 Tanimoto 유사도
    inter = np.atleast_2d(X) @ ref.T
    return (inter / (np.atleast_2d(X).sum(1)[:, None] + ref.sum(1)[None, :] - inter + 1e-9)).max(1)

sur_obj = lambda x: surrogate(x)[0] if valid(x) else -1e9
opt = ga(0, sur_obj)
print(f"최적화된 분자의 조각 수: {int(opt.sum())} (라이브러리는 3개 이하)")
print(f"대리 모델이 예측한 물성: {surrogate(opt)[0]:+.2f}")
print(f"그 분자의 참 물성     : {true_property(opt):+.2f}"
      f"  (예측이 참보다 {surrogate(opt)[0] - true_property(opt):+.2f} 낙관)")
print(f"학습 라이브러리와의 최대 Tanimoto 유사도: {tanimoto_max(opt, lib)[0]:.2f}")

# %% cell 4 — optimizer's curse와 적용 영역 필터
pool = sample[val][n_frag >= 2][:1000]              # 측정 안 한 후보 풀(조각 수 다양)
pred = surrogate(pool)
true = np.array([true_property(x) for x in pool])
sim = tanimoto_max(pool, lib)                       # 각 후보의 라이브러리 유사도

top = np.argsort(pred)[::-1][:20]                   # 예측 상위 20개를 '유망 후보'로 선별
print(f"예측 상위 20개 후보: 평균 예측 {pred[top].mean():+.2f}, 평균 참 물성 {true[top].mean():+.2f}"
      f"  → 예측이 {pred[top].mean() - true[top].mean():+.2f} 낙관(착취)")
thr = np.quantile(sim, 0.7)                         # 적용 영역: 유사도 상위 30%
idx = np.flatnonzero(sim >= thr)
top_dom = idx[np.argsort(pred[idx])[::-1][:20]]
print(f"적용 영역 안에서 상위 20개:  평균 예측 {pred[top_dom].mean():+.2f}, 평균 참 물성 {true[top_dom].mean():+.2f}"
      f"  → 낙관 편향 {pred[top_dom].mean() - true[top_dom].mean():+.2f}")
print(f"예측 상위 20개 중 적용 영역 밖 비율: {(sim[top] < thr).mean():.0%}")
