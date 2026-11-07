"""29.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/life-sciences/bio-evaluation/cells.py
"""
# %% cell 1 — 서열 동일성 군집과 분할
import numpy as np
from scipy.sparse.csgraph import connected_components

rng = np.random.default_rng(0)
L = 80

def evolve(seq, rate):                                 # 각 자리를 rate 확률로 무작위 잔기로 바꾼다
    out = seq.copy()
    hit = rng.random(len(seq)) < rate
    out[hit] = rng.integers(0, 20, hit.sum())
    return out

seqs, sup = [], []
for s in range(15):                                    # 상위 가족 15개
    root = rng.integers(0, 20, L)
    for f in range(4):                                 # 가족 4개씩
        anc = evolve(root, 0.35)
        for _ in range(5):                             # 구성원 5개씩
            seqs.append(evolve(anc, 0.15)); sup.append(s)
S, sup = np.array(seqs), np.array(sup)                 # (300, 80), (300,)
ident = (S[:, None, :] == S[None, :, :]).mean(-1)      # (300, 300) 정렬 없는 서열 동일성

def identity_clusters(ident, t):                       # 동일성 ≥ t로 이어진 서열은 같은 군집(단일 연결)
    return connected_components(ident >= t, directed=False)[1]

def max_id_to_train(test, train):                      # test 서열마다 train에서 가장 가까운 친척
    return ident[np.ix_(test, train)].max(1)

clusters = identity_clusters(ident, 0.3)
random_test = rng.permutation(300) < 60                # 무작위 20%
cl_ids = rng.permutation(clusters.max() + 1)[:int(0.2 * (clusters.max() + 1))]
cluster_test = np.isin(clusters, cl_ids)               # 군집 단위 약 20%
print("동일성 0.3 기준 군집 수:", clusters.max() + 1)
for name, te in [("무작위 분할", random_test), ("군집 분할", cluster_test)]:
    m = max_id_to_train(np.where(te)[0], np.where(~te)[0])
    print(f"{name}: test {te.sum()}개, train과의 최대 동일성 중앙값 {np.median(m):.2f}, 최댓값 {m.max():.2f}")

# %% cell 2 — 성능은 train과의 거리의 함수다
fam = np.repeat(np.arange(60), 5)                      # (300,) 가족 번호
sup_label = rng.integers(0, 2, 15)[np.arange(60) // 4] # (60,) 가족이 속한 상위 가족의 기능
fam_label = np.where(rng.random(60) < 0.15, 1 - sup_label, sup_label)   # 가족의 15%는 기능이 갈라졌다
y = fam_label[fam]                                     # (300,)

def nn_transfer(test, train):                          # 가장 비슷한 train 서열의 label을 옮긴다
    nearest = train[ident[np.ix_(test, train)].argmax(1)]
    return y[nearest] == y[test], max_id_to_train(test, train)

n_cl = clusters.max() + 1
records = {"무작위 분할": [], "가족 분할": [], "군집 분할": []}
for seed in range(5):                                  # 분할을 5번 반복해 test 결과를 모은다
    r = np.random.default_rng(seed)
    splits = {"무작위 분할": r.permutation(300) < 60,
              "가족 분할": np.isin(fam, r.permutation(60)[:12]),
              "군집 분할": np.isin(clusters, r.permutation(n_cl)[:max(1, n_cl // 5)])}
    for name, te in splits.items():
        records[name].append(np.stack(nn_transfer(np.where(te)[0], np.where(~te)[0]), 1))
for name in records:
    records[name] = np.vstack(records[name])           # (n, 2) 정답 여부, train과의 최대 동일성
    print(f"{name}: 정확도 {records[name][:, 0].mean():.2f}")
pooled = np.vstack(list(records.values()))
for lo, hi in [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]:
    sel = (pooled[:, 1] >= lo) & (pooled[:, 1] < hi)
    print(f"  최대 동일성 [{lo:.1f}, {hi:.1f}): {sel.sum():4d}개, 정확도 {pooled[sel, 0].mean():.2f}")

# %% cell 3 — 실험 검증의 성공률과 신뢰구간
from scipy.stats import binomtest, fisher_exact

tested = {"모델 추천": (12, 96), "무작위 선택": (3, 96)}   # (활성이 확인된 수, 실험한 수)
for name, (k, n) in tested.items():
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    print(f"{name}: {k}/{n} = {k / n:.3f}, 95% 신뢰구간 [{ci.low:.3f}, {ci.high:.3f}]")
table = [[12, 96 - 12], [3, 96 - 3]]                   # 2x2 분할표: (성공, 실패) × (모델, 무작위)
print(f"Fisher 정확 검정(단측) p = {fisher_exact(table, alternative='greater')[1]:.3f}")
print(f"30개 중 0개 성공일 때 성공률의 단측 95% 상한: {1 - 0.05 ** (1 / 30):.3f}")

# %% cell 4 — 데이터 버전의 지문
import hashlib

AA = "ACDEFGHIKLMNPQRSTVWY"
ids = [f"P{i:04d}" for i in range(300)]

def fingerprint(ids, S, y):                            # 행 순서와 무관한 데이터셋 지문
    rows = sorted(f"{i}\t{''.join(AA[c] for c in s)}\t{l}" for i, s, l in zip(ids, S, y))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:12]

perm = rng.permutation(300)
y_fixed = y.copy()
y_fixed[17] = 1 - y_fixed[17]                          # label 하나를 고친 판
print("원본            :", fingerprint(ids, S, y))
print("행 순서만 섞음  :", fingerprint([ids[i] for i in perm], S[perm], y[perm]))
print("label 하나 수정 :", fingerprint(ids, S, y_fixed))
