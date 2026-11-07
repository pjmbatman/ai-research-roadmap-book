---
id: python-and-computing/algorithms-and-complexity
title: "자료구조·알고리즘·계산 비용"
title_en: "Data Structures, Algorithms, and Cost"
tier: core
status: draft
prerequisites: [python-and-computing/python-language]
---

# 자료구조·알고리즘·계산 비용

연구 코드는 대개 작은 데이터로 시작한다. 샘플 백 개로 짠 전처리 스크립트는 1초 만에 끝나고, 아무도 그 속도를 의심하지 않는다. 그러다 데이터가 십만 개가 되면 같은 스크립트가 밤새 돈다. 코드가 틀린 것은 아니다. 데이터 크기에 따라 비용이 어떻게 늘어나는지를 고려하지 않았을 뿐이다.

이 장은 알고리즘 교과서의 증명 대신, 연구 코드에서 실제로 시간을 잡아먹는 패턴을 다룬다. Python 내장 자료구조의 연산 비용, 데이터 크기에 따른 시간 증가를 예측하는 복잡도 표기, 메모리에 다 올릴 수 없는 데이터를 다루는 패턴, 희소 표현, 그리고 추측 대신 측정으로 병목을 찾는 profiling이다. 목표는 "느리다"는 느낌을 "이 함수가 이 이유로 $O(n^2)$이고, 고치면 몇 배 빨라진다"는 숫자로 바꾸는 것이다.

## 학습 목표

- 자료구조별 연산 비용을 안다
- 데이터 크기에 따른 병목을 예측한다
- 간단한 profiling을 수행한다

## 자료구조

같은 데이터를 어떤 자료구조에 담느냐에 따라 같은 연산의 비용이 수천 배 달라진다. Python 내장 자료구조의 주요 연산 비용은 다음과 같다($n$은 원소 수, "평균"은 hash 충돌이 드문 일반적인 경우).

| 자료구조 | 구현 | 빠른 연산 | 느린 연산 |
|---|---|---|---|
| `list` | 연속 배열(포인터) | 인덱스 접근 $O(1)$, 끝에 `append`·`pop()` amortized $O(1)$ | `x in lst` $O(n)$, `pop(0)`·`insert(0, x)` $O(n)$ |
| `tuple` | 불변 연속 배열 | 인덱스 접근 $O(1)$, hash 가능(원소가 hash 가능하면) | `x in t` $O(n)$ |
| `dict` | hash table | 키 조회·삽입·삭제 평균 $O(1)$, 삽입 순서 유지 | 값으로 찾기 $O(n)$ |
| `set` | hash table | `x in s`·추가·삭제 평균 $O(1)$, 합·교집합 | 순서 없음, 인덱스 접근 불가 |
| `collections.deque` | 블록 연결 리스트 | 양 끝 넣기·빼기 $O(1)$ | 가운데 인덱스 접근 $O(n)$ |
| `heapq` (list 위) | 이진 heap | 최솟값 보기 $O(1)$, `heappush`·`heappop` $O(\log n)$ | 임의 원소 찾기 $O(n)$ |

**hash table의 직관.** set과 dict는 원소의 hash 값(`hash(x)`)으로 저장 위치를 바로 계산한다. 그래서 원소가 몇 개든 평균적으로 몇 번의 비교로 찾는다. 대가는 두 가지다. 원소가 hash 가능해야 하고(리스트·dict·NumPy 배열은 안 된다. tuple이나 문자열로 바꿔 쓴다), 메모리를 더 쓴다. list의 `in`은 앞에서부터 `==`로 하나씩 비교한다. 비교 횟수를 직접 세어 본다.

```python {exec="browser" session="algo"}
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
```

```output
n=  250  list 비교  31,125회  set 비교 0회
n=  500  list 비교 124,750회  set 비교 0회
n= 1000  list 비교 499,500회  set 비교 0회
set 조회가 100배 이상 빠른가: True
```

중복 없는 $n$개를 list로 거르면 비교가 정확히 $0 + 1 + \cdots + (n-1) = n(n-1)/2$번 일어난다. $n$이 두 배가 될 때마다 비교는 네 배가 된다. set은 hash 값이 다른 원소끼리는 `==`를 부르지도 않아 비교가 0번이다. 시간으로 재도 같은 결론이다. 여기서는 기계마다 다른 시간 값 대신 대소 관계만 출력했다.

**amortized 비용.** list의 `append`는 대부분 $O(1)$이지만, 미리 잡아 둔 공간이 차면 더 큰 공간을 새로 잡고 모든 원소를 옮기는 $O(n)$짜리 재할당이 일어난다. 공간을 일정 비율로 늘리기 때문에 재할당은 점점 드물어지고, $n$번 `append`의 총비용은 $O(n)$이다. 그래서 한 번의 평균 비용, 즉 **amortized** 비용이 $O(1)$이다. 개별 연산은 가끔 느리지만 긴 연산열 전체로는 빠르다는 뜻이다. 반면 `pop(0)`은 매번 나머지 원소를 모두 한 칸씩 옮기므로 amortized로도 $O(n)$이다. 앞에서 꺼내는 대기열은 `deque`로 만든다.

## 복잡도

**왜 필요한가.** 실행 시간은 기계, Python 버전, 다른 프로세스에 따라 달라진다. 그러나 "데이터가 두 배가 되면 시간이 몇 배가 되는가"는 알고리즘이 정한다. 이 증가율을 표현하는 언어가 복잡도다.

**정의.** 입력 크기 $n$에 대한 비용 $T(n)$이 충분히 큰 $n$에서 어떤 상수 $c$에 대해 $T(n) \le c \cdot g(n)$을 만족하면 $T(n) = O(g(n))$이라 쓴다. 상수 배수와 작은 항을 버리고 **증가율**만 남긴 상한이다. 연구 코드에서 만나는 대표적인 증가율은 다음과 같다.

| 복잡도 | 예 | $n$이 10배가 되면 |
|---|---|---|
| $O(1)$ | dict 조회, 배열 인덱스 접근 | 그대로 |
| $O(\log n)$ | 이진 탐색, heap 연산 | 몇 단계 더 |
| $O(n)$ | 한 번 훑기, 합계, set으로 중복 제거 | 10배 |
| $O(n \log n)$ | 정렬 | 10배 남짓 |
| $O(n^2)$ | 모든 쌍 비교, list로 중복 제거 | 100배 |
| $O(n^3)$ | 밀집 행렬 곱, 밀집 선형 시스템 풀이 | 1000배 |

**공간 복잡도**는 추가로 필요한 메모리의 증가율이다. 5.3장의 쌍별 거리를 broadcasting으로 계산하면 시간도 공간도 $O(N^2 D)$이고, 메모리가 먼저 한계에 닿는다.

**직관: 로그의 힘.** 정렬된 데이터에서 이진 탐색은 비교 한 번마다 후보를 절반으로 줄인다. $n$이 백만 배가 되어도 비교는 20번 늘 뿐이다.

```python {exec="browser" session="algo"}
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
```

```output
n=        1,000  bisect가 본 원소 10개  (log2 n ≈ 10)  위치 999
n=    1,000,000  bisect가 본 원소 20개  (log2 n ≈ 20)  위치 999,999
n=1,000,000,000  bisect가 본 원소 30개  (log2 n ≈ 30)  위치 999,999,999
```

::quiz{id="bisect-comparisons"}

**숨은 반복문.** 복잡도를 셀 때 가장 흔한 실수는 한 줄짜리 연산을 $O(1)$로 착각하는 것이다. `x in lst`, `lst.index(x)`, `lst.remove(x)`, `lst.pop(0)`, `s1 + s2`(문자열·리스트 이어 붙이기로 새 객체 만들기), `np.append(arr, x)`(배열 전체 복사), `df.loc[len(df)] = row`(pandas 행 추가)는 모두 내부에서 $O(n)$의 반복을 한다. 이것을 반복문 안에 넣으면 전체가 $O(n^2)$이 된다.

::quiz{id="hidden-quadratic"}

**측정으로 확인하는 복잡도.** 복잡도를 모를 때는 $n$을 두 배씩 늘려 시간을 재고 비율을 본다. 비율이 2에 가까우면 $O(n)$, 4에 가까우면 $O(n^2)$, 8에 가까우면 $O(n^3)$이다. 로그 축 그래프(log–log plot)에서 기울기를 읽어도 같다. 이렇게 얻은 증가율로 실제 데이터 크기에서의 시간을 외삽하면, 밤새 돌릴지 지금 고칠지를 실행 전에 판단할 수 있다.

**연구에서의 의미.** 복잡도는 상수를 버리므로, 작은 $n$에서는 $O(n^2)$ 알고리즘이 $O(n \log n)$보다 빠를 수 있다. 또 NumPy로 벡터화한 $O(n^2)$ 코드가 Python 반복문으로 쓴 $O(n \log n)$ 코드보다 실제 데이터 크기에서 빠른 경우도 흔하다. 복잡도는 "어느 크기에서 무엇이 병목이 될지" 예측하는 도구이고, 최종 판단은 측정이 한다.

## 대용량 처리 패턴

데이터가 메모리보다 크거나, 한 번 훑는 데도 시간이 오래 걸리면 알고리즘 선택이 곧 실행 가능성이 된다. 자주 쓰는 패턴은 다섯 가지다.

- **정렬.** Python의 `sorted`와 `list.sort`는 $O(n \log n)$이고 안정 정렬(같은 키의 원래 순서 유지)이다. 정렬해 두면 탐색($O(\log n)$), 중복 찾기(이웃끼리만 비교), 구간 질의, 두 목록의 병합이 쉬워진다. NumPy는 `np.sort`, `np.argsort`, `np.searchsorted`로 같은 일을 벡터화한다.
- **탐색.** 여러 번 찾을 거라면 한 번 정렬하거나(이진 탐색) dict·set을 만들어 둔다(hash 조회). 한 번만 찾는다면 선형 탐색이 정렬보다 싸다. 공간에서 가까운 점을 찾는 문제(최근접 이웃)는 `scipy.spatial.KDTree` 같은 공간 자료구조를 쓴다.
- **샘플링.** 전체를 볼 필요가 없는 질문(분포 확인, 디버깅, 하이퍼파라미터 탐색의 초기 단계)은 무작위 표본으로 답한다. 길이를 모르는 스트림에서 균등한 표본 $k$개를 뽑는 reservoir sampling은 메모리 $O(k)$로 한 번 훑으며 동작한다. 시드를 고정해 표본을 재현 가능하게 한다.
- **streaming.** 데이터를 한 번에 한 원소(또는 한 조각)씩 읽으며 필요한 요약 통계만 유지한다. 5.1장의 generator가 이 패턴의 도구다.
- **chunking.** 데이터를 메모리에 들어가는 조각으로 나누어 처리하고 결과를 합친다. 조각 안에서는 NumPy로 벡터화하고, 조각 사이만 Python 반복문으로 돈다. 쌍별 거리라면 행을 1000개씩 잘라 `(1000, N)` 블록을 차례로 계산한다.

streaming과 chunking의 핵심은 **조각의 요약을 정확히 합치는 방법**이다. 합계와 개수는 더하면 되지만, 분산은 그렇지 않다. 교과서 공식 $\mathrm{Var}[x] = \mathrm{E}[x^2] - \mathrm{E}[x]^2$를 조각마다 누적하면 수학적으로는 맞지만, 평균이 표준편차보다 훨씬 클 때 비슷한 크기의 두 큰 수를 빼면서 유효 숫자가 사라진다(소거 오차, 9.1장). 안정한 방법은 조각마다 (개수 $n$, 평균 $\bar{x}$, 편차 제곱합 $M_2$)를 구해 다음 식으로 합치는 것이다.

$$
n = n_a + n_b, \quad \delta = \bar{x}_b - \bar{x}_a, \quad \bar{x} = \bar{x}_a + \delta \frac{n_b}{n}, \quad M_2 = M_{2,a} + M_{2,b} + \delta^2 \frac{n_a n_b}{n}
$$

분산은 $M_2 / n$이다. 원소 하나씩 갱신하는 특수한 경우가 Welford 방법으로 알려진 갱신식이다.

```python {exec="browser" session="algo"}
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
```

```output
평균 일치: True  분산 일치: True
실제 분산: 1.0013  교과서 공식의 오차가 0.5보다 큰가: True
```

조각을 합친 결과는 전체 데이터로 계산한 값과 일치하고, 교과서 공식은 분산 1 정도의 데이터에서 0.5 이상 틀렸다. $x^2 \approx 10^{16}$ 근처에서 `float64`가 구별할 수 있는 간격은 약 2이므로, 그보다 작은 분산은 두 큰 수의 차이에 묻힌다. 측정 시각(Unix 시간 초), 절대 온도, 해발 고도처럼 기준점에서 멀리 떨어진 값이 이런 데이터다.

:::note[분야 예시]
천문 관측 survey나 유전체 시퀀싱은 원시 데이터가 수십 TB에 이른다. 전체를 한 번에 읽는 대신 파일 단위로 streaming하며 천체·유전자별 요약 통계를 갱신하고, 조각별 결과를 위 공식으로 합친다. 조각을 서로 다른 노드에서 병렬로 처리해도 결과가 같다는 점이 이 방식의 장점이다. 합치는 연산이 결합법칙을 만족하기 때문이다.
:::

## 희소와 밀집

**왜 필요한가.** 그래프의 인접 행렬, 유한차분·유한요소의 계수 행렬, 문서–단어 행렬, 단백질 상호작용 네트워크는 대부분의 원소가 0이다. 이를 밀집 배열로 저장하면 0을 저장하는 데 메모리를 쓰고, 0을 곱하는 데 시간을 쓴다.

**정의.** 희소(sparse) 표현은 0이 아닌 원소의 값과 위치만 저장한다. `scipy.sparse`의 대표 형식은 세 가지다.

- **COO**(coordinate): (행, 열, 값) 세 배열. 만들기 쉽고, 중복 위치는 변환할 때 더해진다.
- **CSR**(compressed sparse row): 행마다 0이 아닌 원소를 모아 저장한다. 값 배열과 열 인덱스 배열(길이 nnz), 각 행의 시작 위치 배열(길이 $n+1$)로 이루어진다. 행 slicing과 행렬–벡터 곱이 빠르다.
- **CSC**: CSR의 열 버전. 열 slicing이 빠르다.

CSR의 메모리는 대략 $\text{nnz} \times (\text{값 크기} + \text{인덱스 크기}) + (n+1) \times \text{인덱스 크기}$이고, 밀집 행렬은 $n^2 \times \text{값 크기}$다.

```python {exec="browser" session="algo"}
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
```

```output
0이 아닌 원소 999,946개  희소 12.4 MB  밀집 80 GB
희소·밀집 행렬-벡터 곱 일치: True
```

같은 행렬이 희소 형식으로 12.4 MB, 밀집 형식으로 80 GB다. 행렬–벡터 곱의 비용도 $O(n^2)$에서 $O(\text{nnz})$로 줄어든다.

**언제 유리한가.** 희소 표현은 공짜가 아니다. 원소마다 인덱스를 함께 저장하고, 메모리 접근이 불규칙해 원소당 계산은 밀집보다 느리다. 대략적인 기준은 다음과 같다.

- 0이 아닌 비율(밀도)이 수 % 이하이고, 행렬이 커서 밀집 형식이 메모리에 부담일 때 희소가 유리하다. 밀도가 수십 %라면 밀집 쪽이 대개 빠르다.
- 희소 행렬에 하는 연산이 희소성을 유지해야 한다. 행렬–벡터 곱, 희소 행렬끼리의 곱, 희소 선형 시스템 풀이(`scipy.sparse.linalg`)는 유지한다. 역행렬(`inv`)은 대개 밀집이 되므로 피하고, 선형 시스템은 `spsolve`나 반복법으로 직접 푼다(9.3장).
- 형식마다 빠른 연산이 다르다. 원소를 하나씩 추가하며 만들 때는 COO나 LIL로 만든 뒤 CSR로 바꾸어 계산한다. CSR에 원소를 하나씩 넣으면 매번 구조 전체를 고쳐야 해서 느리다.

## profiling 기초

**왜 필요한가.** 병목은 거의 항상 예상하지 못한 곳에 있다. "이 수치 계산이 느릴 것"이라고 생각하고 최적화했는데, 실제로는 샘플마다 같은 파일을 다시 읽는 코드가 시간의 대부분을 쓰는 일이 흔하다. 측정 없이 최적화하면 빠르게 하지 않아도 되는 코드를 복잡하게 만들고, 진짜 병목은 그대로 남는다.

**도구.** 측정 범위가 좁은 것부터 넓은 것 순서로 정리한다.

- **`time.perf_counter()`**: 구간 하나의 경과 시간. 해상도가 높고 단조 증가하는 시계다. 한 번의 측정은 잡음이 크므로 여러 번 재서 최솟값이나 중앙값을 쓴다.
- **`timeit`**: 짧은 코드 조각을 여러 번 반복 실행해 시간을 잰다. 명령행 `python -m timeit "..."`이나 IPython의 `%timeit`. 마이크로 벤치마크용이다.
- **`cProfile`**: 표준 라이브러리의 결정적(deterministic) profiler. 프로그램이 도는 동안 모든 함수 호출을 기록해, 함수별 호출 횟수(`ncalls`), 그 함수 자체에서 쓴 시간(`tottime`), 그 함수가 부른 함수까지 포함한 시간(`cumtime`)을 보고한다. `python -m cProfile -s cumtime script.py`로 스크립트 전체를 측정할 수 있다. 함수 호출마다 부하가 붙어 절대 시간은 부풀려지지만, 호출 횟수는 정확하다.
- **line profiler**: 함수 안에서 **줄마다** 시간을 잰다. cProfile이 "어느 함수"를 알려 준다면, line profiler는 "그 함수의 어느 줄"을 알려 준다. `line_profiler` 같은 외부 패키지를 쓴다.
- **메모리 측정**: 표준 라이브러리 `tracemalloc`은 Python 수준의 메모리 할당을 추적해 최대 사용량과 할당 위치를 보고한다. 줄 단위 메모리 사용량을 보는 외부 패키지(`memory_profiler` 등)도 있다. 외부 패키지는 유지 상태와 사용법이 바뀔 수 있으므로 설치 전에 공식 저장소에서 확인한다.

cProfile은 브라우저 실행 환경에서도 쓸 수 있다. 샘플마다 라벨 파일 전체를 다시 읽는 작은 로더를 측정해, 결정적인 값인 호출 횟수만 출력한다.

```python {exec="browser" session="algo"}
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
```

```output
{'get_item': 500, 'loads': 500, 'read_labels': 500}
```

`json.loads`가 샘플 수만큼 500번 불렸다. 라벨 파일은 하나이므로 한 번이면 충분하다. profiling 결과를 읽을 때는 "어느 함수가 느린가"보다 먼저 "**호출 횟수가 기대와 맞는가**"를 본다. 기대보다 많이 불린 함수는 반복 작업이 새는 곳이다.

::quiz{id="profile-reading"}

**측정의 원칙.**

1. **실제 크기에 가까운 입력으로 잰다.** 작은 입력에서는 $O(n^2)$ 부분이 보이지 않는다.
2. **전체를 먼저, 부분을 나중에.** 프로그램 전체를 cProfile로 보고 `cumtime`이 큰 함수를 찾은 뒤, 그 함수 안을 line profiler로 본다.
3. **고치기 전과 후를 같은 조건에서 잰다.** 결과가 같은지(수치 일치)도 함께 확인한다. 빨라졌지만 답이 달라졌다면 개선이 아니다.
4. **GPU 코드는 따로 다룬다.** GPU 연산은 비동기로 실행되어, CPU 쪽 시계로 재면 실제 계산 시간이 아닌 다른 것을 잴 수 있다(5.5장, 22.4장).

## 실습

데이터 로더의 반복 파일 읽기와 중복 탐색 병목을 profiling으로 찾아 개선한다. 코드는 `labs/python-and-computing/algorithms-and-complexity/loader_bottleneck.py`다. 임시 디렉터리에 샘플 파일 3000개(`.npy`)와 라벨 파일 하나(JSON)를 만들고, 중복이 섞인 샘플 ID 목록 6000개로 batch를 만든다. 느린 버전에는 두 병목이 숨어 있다.

1. 샘플마다 라벨 파일 전체를 다시 읽고 파싱한다(반복 파일 읽기).
2. 이미 본 샘플 ID를 리스트에 기록하고 membership을 검사한다($O(n^2)$ 중복 탐색).

이 기계에서의 예는 다음과 같다. 시간 값은 기계마다 다르다.

```output
== cProfile: slow_loader, tottime 상위 5개 ==
         502474 function calls (479164 primitive calls) in 2.328 seconds

   Ordered by: internal time
   List reduced from 99 to 5 due to restriction <5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     2590    1.443    0.001    1.443    0.001 decoder.py:351(raw_decode)
        1    0.179    0.179    2.328    2.328 loader_bottleneck.py:33(slow_loader)
     2590    0.067    0.000    0.067    0.000 {built-in method builtins.compile}
     5180    0.058    0.000    0.113    0.000 {built-in method _io.open}
     2590    0.040    0.000    0.044    0.000 {built-in method numpy.fromfile}


결과 동일: True  샘플 2590개
slow 1.592 s  →  fast 0.125 s  (13배)
```

결과를 다음 순서로 읽는다.

- **첫 줄이 병목 1이다.** JSON 파싱(`raw_decode`)이 고유 샘플 수(2590)만큼 불려 전체 시간의 60% 이상을 썼다. 라벨 파일을 한 번만 읽으면 사라진다.
- **병목 2는 함수로 보이지 않는다.** 리스트의 `in`은 별도 함수 호출이 아니라서, 그 비용이 `slow_loader` 자신의 `tottime`(0.179 s)에 섞여 있다. cProfile은 함수 단위 도구라는 한계가 여기서 드러난다. 이런 비용은 line profiler로 줄 단위로 보거나, 이 장에서처럼 연산의 복잡도로 추론한다.
- **예상하지 못한 항목도 있다.** `compile`이 2590번 불렸다. `np.load`가 `.npy` 파일 머리의 메타데이터를 읽는 과정에서 부르는 것이다. 작은 파일을 수천 개 읽는 구조 자체의 비용이고, 샘플을 큰 파일 몇 개로 묶는 저장 형식(15.2장)이 근본적인 해결이다.
- **보고는 숫자로 한다.** 개선 전후의 시간, 배율, 그리고 결과가 같다는 확인을 함께 적는다.

`N_SAMPLES`를 두 배로 늘려 느린 버전과 빠른 버전의 시간이 각각 몇 배가 되는지 예측하고 실행해 확인한다. 이어서 두 연습문제를 푼다. 테스트가 바로 채점한다.

::exercise{id="dedup-linear"}

::exercise{id="streaming-stats"}

## 실패 사례

- **list에서 반복적으로 membership을 검사하는 것.** `if x not in seen_list`를 반복문 안에 두면 전체가 $O(n^2)$이다. 데이터 1000개에서는 보이지 않다가 10만 개에서 만 배 느려진다. 이미 본 값은 set이나 dict에 기록한다. 같은 이유로 반복문 안의 `lst.index`, `lst.remove`, `pop(0)`, `np.append`, pandas 행 추가를 의심한다.
- **측정 없이 최적화하는 것.** 느릴 것 같은 수치 계산을 C 확장이나 병렬 처리로 바꾸는 데 며칠을 쓰고, 실제 병목이었던 반복 파일 읽기는 그대로 두는 일이 흔하다. 먼저 cProfile로 전체를 재고, 시간의 대부분을 쓰는 곳만 고친다.
- **작은 입력으로 성능을 판단하는 것.** 샘플 100개로 1초면 10만 개로 1000초($O(n)$)일 수도, 100만 초($O(n^2)$)일 수도 있다. $n$을 두 배씩 늘려 증가율을 확인한 뒤 외삽한다.
- **분산을 $\mathrm{E}[x^2] - \mathrm{E}[x]^2$로 누적하는 것.** streaming 처리에서 흔히 쓰지만, 평균이 큰 데이터에서 소거 오차로 무너진다. 조각별 (개수, 평균, 편차 제곱합)을 합치는 안정한 공식을 쓴다.
- **희소 행렬을 무심코 밀집으로 바꾸는 것.** `A.toarray()`나 역행렬처럼 결과가 밀집이 되는 연산은 메모리를 폭발시킨다. `A.mean(axis=0)` 같은 축소 연산의 결과도 밀집 배열이지만 크기가 작으니 괜찮다. 큰 희소 행렬에서는 결과의 형식을 확인한다.

## 통과 기준

- 자신의 연구 코드(또는 `loader_bottleneck.py`)에서 cProfile로 병목을 찾고, 개선 전후의 시간·배율과 결과 일치 여부를 숫자로 보고한다.
- $n$을 두 배씩 늘려 측정한 시간 비율로 코드 한 부분의 복잡도를 추정하고, 실제 데이터 크기에서의 시간을 예측한다.
- 연습문제 두 개(`dedup-linear`, `streaming-stats`)의 테스트를 통과한다.

## 참고 문헌

- Python Software Foundation. *The Python Standard Library* — `collections.deque`, `heapq`, `bisect`, `timeit`, The Python Profilers(`cProfile`, `pstats`), `tracemalloc`. Python 공식 문서.
- SciPy 공식 문서 — Sparse arrays(`scipy.sparse`)의 저장 형식과 연산. scipy.org.
