"""5.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/python-and-computing/algorithms-and-complexity/cells.py
시간 측정값은 기계마다 다르므로, 비교 횟수처럼 결정적인 양이나 대소 관계만 출력한다.
"""
# %% cell 1 — list와 set의 membership: 비교 횟수와 시간
import time

class Key:
    """== 비교가 몇 번 일어나는지 세는 값"""
    eq_calls = 0
    def __init__(self, v): self.v = v
    def __eq__(self, other):
        Key.eq_calls += 1
        return self.v == other.v
    def __hash__(self): return hash(self.v)

for n in [250, 500, 1_000]:
    items = [Key(i) for i in range(n)]
    Key.eq_calls = 0
    seen_list = []
    for k in items:
        if k not in seen_list:       # 리스트 전체를 훑는다
            seen_list.append(k)
    list_calls = Key.eq_calls
    Key.eq_calls = 0
    seen_set = set()
    for k in items:
        if k not in seen_set:        # hash로 바로 찾아간다
            seen_set.add(k)
    print(f"n={n:5d}  list 비교 {list_calls:>7,d}회  set 비교 {Key.eq_calls}회")

data = list(range(50_000)); data_set = set(data); queries = range(49_500, 50_000)
t0 = time.perf_counter(); sum(q in data for q in queries); t1 = time.perf_counter()
sum(q in data_set for q in queries); t2 = time.perf_counter()
print("set 조회가 100배 이상 빠른가:", (t1 - t0) > 100 * (t2 - t1))

# %% cell 2 — 정렬된 데이터의 이진 탐색: 비교 횟수
import bisect

calls = 0
def key(v):                                         # bisect가 원소를 볼 때마다 불린다
    global calls
    calls += 1
    return v

for n in [1_000, 1_000_000, 1_000_000_000]:
    arr = range(0, 2 * n, 2)                       # 정렬된 짝수 n개(메모리에 만들지 않는 sequence)
    calls = 0
    pos = bisect.bisect_left(arr, 2 * n - 2, key=key)   # 마지막 원소 찾기
    print(f"n={n:>13,d}  bisect가 본 원소 {calls}개  (log2 n ≈ {n.bit_length()})  위치 {pos:,d}")

# %% cell 3 — streaming: 조각(chunk)별 통계를 합치기
import numpy as np

def merge_stats(a, b):
    """(개수, 평균, 편차제곱합) 두 묶음을 합친다(Chan 등의 병렬 분산 공식)."""
    na, ma, m2a = a
    nb, mb, m2b = b
    n = na + nb
    delta = mb - ma
    return n, ma + delta * nb / n, m2a + m2b + delta**2 * na * nb / n

rng = np.random.default_rng(0)
x = 1e8 + rng.normal(size=1_000_000)               # 평균이 큰 측정값 (N,)
total = (0, 0.0, 0.0)
for chunk in np.array_split(x, 37):                # 한 번에 조각 하나만 본다고 하자
    c = (len(chunk), chunk.mean(), ((chunk - chunk.mean()) ** 2).sum())
    total = merge_stats(total, c)
n, mean, m2 = total
print("평균 일치:", np.isclose(mean, x.mean()), " 분산 일치:", np.isclose(m2 / n, x.var()))
naive = (x**2).mean() - x.mean() ** 2              # E[x^2] - E[x]^2: 소거 오차
print("실제 분산:", round(float(x.var()), 4), " 교과서 공식의 오차가 0.5보다 큰가:", abs(naive - x.var()) > 0.5)

# %% cell 4 — 희소와 밀집: 저장 비용
from scipy import sparse

n, k = 100_000, 1_000_000                          # 행마다 0이 아닌 원소 약 10개
rows = rng.integers(0, n, size=k, dtype=np.int32)
cols = rng.integers(0, n, size=k, dtype=np.int32)
A = sparse.csr_array((rng.normal(size=k), (rows, cols)), shape=(n, n))   # 중복 위치는 더해진다
sparse_bytes = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
dense_bytes = n * n * 8                            # float64 밀집 행렬이라면
print(f"0이 아닌 원소 {A.nnz:,d}개  희소 {sparse_bytes / 1e6:.1f} MB  밀집 {dense_bytes / 1e9:.0f} GB")

v = rng.normal(size=n)                             # (n,)
small = A[:2000, :2000]
print("희소·밀집 행렬-벡터 곱 일치:", np.allclose(small @ v[:2000], small.toarray() @ v[:2000]))

# %% cell 5 — cProfile로 반복 파일 읽기 병목 찾기
import cProfile
import json
import pstats
import tempfile
from pathlib import Path

meta_path = Path(tempfile.mkdtemp()) / "labels.json"
meta_path.write_text(json.dumps({str(i): i % 3 for i in range(2_000)}))

def read_labels():
    return json.loads(meta_path.read_text())

def get_item(i):
    return i, read_labels()[str(i)]                # 샘플마다 파일 전체를 다시 읽는다

prof = cProfile.Profile()
prof.enable()
batch = [get_item(i) for i in range(500)]
prof.disable()
stats = pstats.Stats(prof).stats                   # {(파일, 줄, 함수): (원시 호출, 호출, ...)}
calls = {fn[2]: s[1] for fn, s in stats.items() if fn[2] in ("get_item", "read_labels", "loads")}
print(dict(sorted(calls.items())))
