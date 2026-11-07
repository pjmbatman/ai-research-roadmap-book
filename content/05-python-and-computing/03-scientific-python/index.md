---
id: python-and-computing/scientific-python
title: "NumPy·pandas·SciPy"
title_en: "Scientific Python: NumPy, pandas, SciPy"
tier: core
status: draft
prerequisites: [python-and-computing/python-language]
---

# NumPy·pandas·SciPy

과학 계산 코드의 대부분은 배열 위에서 일어난다. 측정값 행렬, 시뮬레이션 격자, 이미지, 시계열, 그리고 나중에 다룰 PyTorch 텐서까지 모두 "같은 타입의 수를 일정한 간격으로 늘어놓은 메모리"라는 같은 구조를 공유한다. NumPy는 이 구조를 Python에서 다루는 표준이고, pandas는 그 위에 이름 붙은 행·열과 시간 인덱스를, SciPy는 최적화·보간·적분·희소 행렬 같은 수치 알고리즘을 얹는다.

이 장은 함수 목록을 외우는 대신, 결과를 조용히 틀리게 만드는 세 가지 동작을 정확히 다룬다. **view와 copy**(원본이 바뀌는가), **dtype**(값이 잘리거나 넘치는가), **broadcasting**(shape이 의도대로 맞춰졌는가)이다. 이 세 가지를 정확히 다루면, 같은 계산을 반복문 없이 벡터화해 수십~수백 배 빠르게 쓰는 일도 자연스럽게 따라온다. 5.1장의 참조 모델(이름표와 객체)이 배열에서 어떻게 확장되는지 보는 장이기도 하다.

## 학습 목표

- 배열의 shape·axis·broadcasting을 정확히 다룬다
- 벡터화로 계산을 표현한다
- 표 데이터와 시간 데이터를 처리한다

## NumPy 배열

Python 리스트로 1000 × 1000 행렬을 만들면 리스트 1000개 안에 float **객체** 백만 개가 흩어져 있다. 원소마다 타입 정보와 참조 횟수를 가진 객체이고, 리스트는 그 객체를 가리키는 포인터를 담을 뿐이다. 이런 구조에서는 원소 하나를 더할 때마다 포인터를 따라가 타입을 확인해야 해서 느리고, C나 Fortran으로 쓴 수치 라이브러리(BLAS, LAPACK)에 넘길 수도 없다.

**정의.** NumPy 배열(`np.ndarray`)은 다음 네 가지로 정의된다.

- **데이터 버퍼**: 같은 타입의 원소가 빈틈없이 놓인 연속된 메모리 블록
- **dtype**: 원소 하나의 타입과 크기. `float64`(8바이트), `float32`(4바이트), `int64`, `uint8`, `bool` 등
- **shape**: 각 축(axis)의 길이. `(N, D)`는 축 0의 길이가 $N$, 축 1의 길이가 $D$
- **strides**: 각 축 방향으로 한 칸 움직일 때 메모리에서 건너뛸 바이트 수

shape과 strides가 "버퍼를 어떻게 읽을지"를 정한다. 원소 `x[i, j]`의 주소는

$$
\text{주소}(i, j) = \text{시작 주소} + i \cdot s_0 + j \cdot s_1
$$

이다. 여기서 $s_0, s_1$이 strides다. 기본 배치(C order, row-major)에서는 마지막 축이 메모리에서 연속이다. `float64`인 `(3, 4)` 배열의 strides는 `(32, 8)`로, 한 행을 건너려면 32바이트(원소 4개), 한 열을 건너려면 8바이트를 움직인다.

**dtype.** 배열의 dtype은 만들 때 정해지고, 원소를 할당해도 바뀌지 않는다. 그래서 값이 dtype에 맞지 않으면 NumPy는 오류 대신 **변환**한다.

```python {exec="browser" session="sci"}
import numpy as np

a = np.array([1, 2, 3], dtype=np.int64)  # 정수 배열
a[0] = 2.9                               # float를 넣어도 정수로 잘린다
print("a =", a, a.dtype)

counts = np.array([200, 100], dtype=np.uint8)
print("uint8 합:", counts + counts)      # 255를 넘으면 256으로 나눈 나머지

big = np.float32(16_777_216)             # 2**24
print("float32: 2**24 + 1 == 2**24 ?", big + np.float32(1) == big)
print("평균 계산 dtype:", np.mean(np.array([1, 2], dtype=np.int64)).dtype)
```

```output
a = [2 2 3] int64
uint8 합: [144 200]
float32: 2**24 + 1 == 2**24 ? True
평균 계산 dtype: float64
```

세 가지 함정이 모두 오류 없이 지나갔다.

- **정수 배열에 실수를 넣으면 잘린다.** `np.array([1, 2, 3])`처럼 정수 리터럴로 만든 배열은 정수 dtype이다. 나중에 실수 결과를 담을 배열은 `np.zeros(n)`(기본 `float64`)이나 `dtype=float`로 만든다.
- **고정 폭 정수는 넘친다(overflow).** `uint8`의 범위는 0–255이고, 넘치면 256으로 나눈 나머지가 된다. 8비트 이미지의 픽셀을 더하거나 평균을 낼 때 먼저 `astype(np.float32)`로 바꾸는 이유다.
- **`float32`는 정수도 $2^{24}$까지만 정확히 표현한다.** 유효 숫자가 약 7자리다. 누적 합이나 큰 인덱스를 `float32`로 다루면 작은 값이 더해지지 않고 사라진다. 부동소수점의 구조는 5.5장과 9.1장에서 다룬다.

평균처럼 결과가 실수인 축소 연산은 정수 입력에 대해 `float64`로 계산한다. 반면 `+`, `*` 같은 원소별 연산은 입력의 dtype을 유지한다.

**axis.** `X.sum(axis=0)`의 `axis=0`은 "축 0을 따라 더해서 **없앤다**"는 뜻이다. shape `(N, D)`에서 축 0을 없애면 `(D,)`, 즉 특성마다의 합이다. "남길 축"으로 읽으면 반대로 틀린다. `keepdims=True`를 주면 없앤 축을 길이 1로 남겨 `(1, D)`가 되고, 이후의 broadcasting이 의도대로 맞춰진다.

## indexing

인덱싱은 세 종류이고, 결과가 **view인가 copy인가**가 종류마다 정해져 있다.

- **basic slicing** — `x[1:3]`, `x[:, 0]`, `x[::2]`, `x[..., None]`. 정수와 slice만 쓴다. 결과는 **view**다. 새 데이터를 만들지 않고, 같은 버퍼를 다른 시작 주소·shape·strides로 본다. `reshape`, `transpose`(`.T`)도 가능하면 view를 만든다.
- **fancy indexing** — `x[[0, 2]]`, `x[idx_array]`. 정수 배열로 고른다. 결과는 항상 **copy**다. 고른 원소들이 일정한 stride로 놓여 있다는 보장이 없기 때문이다.
- **boolean mask** — `x[x > 5]`. 같은 shape의 bool 배열로 고른다. 결과는 조건을 만족한 원소를 모은 1차원 **copy**다.

view는 5.1장의 이름표와 같은 구조다. 원본과 view는 같은 메모리를 가리키는 두 개의 배열 객체이고, 한쪽을 제자리에서 바꾸면 다른 쪽에서도 보인다.

```python {exec="browser" session="sci"}
x = np.arange(12, dtype=np.float64).reshape(3, 4)   # (3, 4)
print("shape:", x.shape, " strides(byte):", x.strides)
print("x.T strides:", x.T.strides, " C 연속인가:", x.T.flags.c_contiguous)

col = x[:, 1]                    # slicing → view: 같은 메모리를 다른 방식으로 본다
col[0] = 100.0
print("view 수정 뒤 x[0, 1] =", x[0, 1], " 메모리 공유:", np.shares_memory(col, x))

picked = x[[0, 2]]               # fancy indexing → copy
picked[:] = -1.0
mask = x > 5                     # boolean mask → copy
print("copy 수정 뒤 x[0] =", x[0], " 메모리 공유:", np.shares_memory(picked, x))
print("mask로 고른 원소 수:", x[mask].size, " shape:", x[mask].shape)
```

```output
shape: (3, 4)  strides(byte): (32, 8)
x.T strides: (8, 32)  C 연속인가: False
view 수정 뒤 x[0, 1] = 100.0  메모리 공유: True
copy 수정 뒤 x[0] = [  0. 100.   2.   3.]  메모리 공유: False
mask로 고른 원소 수: 7  shape: (7,)
```

전치 `x.T`는 데이터를 옮기지 않고 strides만 `(32, 8)`에서 `(8, 32)`로 바꾼 view다. 그래서 공짜지만, 마지막 축이 더는 메모리에서 연속이 아니다(C 연속이 아님). 연속성은 성능에 영향을 준다(5.5장). 열 `x[:, 1]`도 view이고, 그것을 바꾸자 원본의 `x[0, 1]`이 100이 되었다. fancy indexing으로 꺼낸 `picked`는 아무리 바꿔도 원본과 무관하다.

::quiz{id="which-modify-original"}

**할당문의 왼쪽은 다르다.** `x[x > 5] = 0`이나 `x[[0, 2]] = -1`처럼 인덱싱이 할당문의 왼쪽에 있으면, copy를 만드는 것이 아니라 원본의 해당 위치에 직접 쓴다. "꺼낸 결과가 view인가"와 "그 위치에 쓰는가"는 다른 질문이다.

**연구에서의 의미.** 함수가 배열을 받아 일부를 잘라 정규화하는 코드, 예를 들어 `ref = data[:, :100]; ref -= ref.mean()`은 호출한 쪽의 `data`를 바꾼다. 같은 데이터를 다른 실험이 다시 쓰면 이미 정규화된 데이터를 한 번 더 정규화한다. 원본을 보존해야 하면 `.copy()`를 명시하고, 의심스러우면 `np.shares_memory(a, b)`로 확인한다. PyTorch 텐서의 slicing·`view`·`transpose`도 같은 규칙을 따른다(17.2장).

## broadcasting과 벡터화

**왜 필요한가.** 행렬의 각 행에서 평균 벡터를 빼려면, 평균 벡터를 행 수만큼 복사한 행렬을 만들어 빼면 된다. 그러나 이는 메모리 낭비다. broadcasting은 shape이 다른 배열 사이의 원소별 연산을 **복사 없이** 정의하는 규칙이다.

**규칙.** 두 배열의 shape을 **뒤에서부터** 축 하나씩 비교한다.

1. 축의 수가 다르면, 적은 쪽의 앞에 길이 1인 축을 붙여 맞춘다. `(D,)`는 `(1, D)`로 취급된다.
2. 각 축에서 두 길이가 같거나, 한쪽이 1이어야 한다. 아니면 `ValueError`다.
3. 길이가 1인 축은 상대의 길이만큼 늘어난 것처럼 취급된다(실제로는 stride 0으로 같은 원소를 반복해서 읽는다).

예를 들어 `(N, D) - (D,)`는 `(N, D) - (1, D)`가 되어 각 행에서 같은 벡터를 뺀다. `(N, 1) * (1, M)`은 `(N, M)`의 바깥 곱(outer product)이다.

**직관과 함정.** 이 규칙은 너무 관대하다. shape이 **틀렸는데도** 규칙상 호환되면 조용히 큰 배열을 만든다. 가장 흔한 사고는 열벡터 `(N, 1)`과 1차원 배열 `(N,)`의 연산이다.

```python {exec="browser" session="sci"}
rng = np.random.default_rng(0)
N = 5
y_true = rng.normal(size=N)                 # (N,)
y_pred = y_true[:, None] + 0.1              # (N, 1) — 모델이 열벡터로 반환했다고 하자
diff = y_pred - y_true                      # (N, 1) - (N,) → (N, N)!
print("diff shape:", diff.shape)
print("잘못된 MSE:", round(float(np.mean(diff**2)), 4),
      " 올바른 MSE:", round(float(np.mean((y_pred[:, 0] - y_true)**2)), 4))

P = rng.normal(size=(200, 3))               # (N, D) 입자 좌표
def pairwise_loop(P):
    n = len(P)
    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(np.sum((P[i] - P[j]) ** 2))
    return D

def pairwise_vec(P):
    d = P[:, None, :] - P[None, :, :]       # (N, 1, D) - (1, N, D) → (N, N, D)
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d))   # (N, N)

import time
t0 = time.perf_counter(); D1 = pairwise_loop(P); t1 = time.perf_counter()
D2 = pairwise_vec(P); t2 = time.perf_counter()
print("결과 일치:", np.allclose(D1, D2), " 벡터화가 20배 이상 빠른가:", (t1 - t0) > 20 * (t2 - t1))
```

```output
diff shape: (5, 5)
잘못된 MSE: 0.3032  올바른 MSE: 0.01
결과 일치: True  벡터화가 20배 이상 빠른가: True
```

예측이 정답보다 정확히 0.1씩 크므로 MSE는 $0.1^2 = 0.01$이어야 한다. 그런데 shape이 어긋난 식은 25개 쌍 전체의 차이를 평균해 0.3032를 냈다. 오류 메시지는 없다. 학습 loss로 이 값을 쓰면 모델은 "모든 정답의 평균"을 맞추도록 학습되어, 예측이 평균 근처로 뭉친다. 방어책은 손실을 계산하기 직전에 `assert y_pred.shape == y_true.shape`를 두는 것이다.

::quiz{id="broadcast-column-minus-row"}

**벡터화.** 두 번째 부분은 같은 broadcasting을 **의도적으로** 쓴 예다. 입자 $N$개의 모든 쌍 거리

$$
D_{ij} = \lVert \mathbf{p}_i - \mathbf{p}_j \rVert = \sqrt{\textstyle\sum_k (p_{ik} - p_{jk})^2}
$$

를 이중 반복문 대신 `(N, 1, D) - (1, N, D) → (N, N, D)` 하나로 계산했다. 결과는 같고(`allclose`), 시간은 이 셀을 실행한 기계에서 20배 이상 차이가 났다(실제 배율은 기계마다 다르므로 대소 관계만 출력한다). 벡터화가 빠른 이유는 계산량이 줄어서가 아니다. 반복문은 원소마다 Python 인터프리터의 비용(바이트코드 해석, 객체 생성, 타입 확인)을 치르고, 벡터화된 연산은 반복을 컴파일된 코드 안에서 연속 메모리 위로 돌린다.

벡터화에는 대가도 있다. 중간 배열 `d`는 $N \times N \times D$개의 원소를 가진다. $N = 10^5$이면 `float64`로 수백 GB가 되어 메모리에 들어가지 않는다. 이때는 전개식 $\lVert\mathbf{x} - \mathbf{y}\rVert^2 = \lVert\mathbf{x}\rVert^2 - 2\mathbf{x}^\top\mathbf{y} + \lVert\mathbf{y}\rVert^2$로 행렬 곱 하나에 맡기거나(연습문제), 행을 나누어(chunking, 5.4장) 계산한다.

**einsum.** `np.einsum("ijk,ijk->ij", d, d)`는 첨자 표기로 연산을 적는다. 입력의 첨자 `ijk`와 출력의 첨자 `ij`를 비교해, 출력에 없는 `k`에 대해 곱을 더한다. 즉 $\sum_k d_{ijk} d_{ijk}$다. 행렬 곱은 `"ik,kj->ij"`, batch 행렬 곱은 `"bik,bkj->bij"`, 대각합은 `"ii->"`다. 축의 의미를 식에 그대로 적기 때문에 shape 실수를 줄인다. 텐서 연산으로서의 einsum은 7.7장에서 자세히 다룬다.

:::note[분야 예시]
분자 동역학이나 천체 N-체 시뮬레이션의 핵심은 모든 입자 쌍 사이의 힘을 계산하는 것이다. 수백 개 입자까지는 위와 같은 `(N, N, D)` broadcasting으로 충분히 빠르다. 입자가 수만 개를 넘으면 $O(N^2)$ 메모리와 계산 자체가 문제가 되어, 일정 거리(cutoff) 안의 이웃만 찾는 자료구조나 희소 표현으로 넘어간다(5.4장).
:::

## pandas

**왜 필요한가.** 실험 데이터는 순수한 수 행렬로 오지 않는다. 열마다 이름과 타입이 다르고(관측소 이름, 시각, 온도, 품질 플래그), 행에는 의미 있는 식별자(시각, 샘플 ID)가 붙어 있으며, 여러 표를 키로 이어 붙여야 한다. pandas는 이런 **표 데이터**(tabular data)를 위한 라이브러리다.

**정의.** `Series`는 **index**가 붙은 1차원 배열, `DataFrame`은 같은 index를 공유하는 이름 붙은 열들의 묶음이다. 각 열은 내부적으로 NumPy 배열(또는 Arrow 배열)이고 열마다 dtype이 다를 수 있다. pandas의 핵심 동작은 **index 정렬**(alignment)이다. 두 Series를 더하면 위치가 아니라 index 값이 같은 원소끼리 더하고, 한쪽에만 있는 index는 `NaN`이 된다. NumPy와 달리 "같은 위치"가 아니라 "같은 이름"으로 짝짓는 것이다.

자주 쓰는 연산은 다음과 같다.

- **groupby**: 열 값으로 행을 묶고(split), 묶음마다 함수를 적용하고(apply), 결과를 합친다(combine). `df.groupby("station")["temp_c"].mean()`
- **merge**: 키 열로 두 표를 잇는다. `how="left"`는 왼쪽 표의 행을 모두 남기고, `how="inner"`는 양쪽에 모두 있는 키만 남긴다.
- **pivot / pivot_table**: 긴(long) 형식을 넓은(wide) 형식으로 바꾼다. 행은 시각, 열은 관측소, 값은 온도인 표가 그 예다. 반대는 `melt`다.

**시간 데이터.** index가 `DatetimeIndex`이면 시간 연산이 생긴다. `resample("D")`는 시간을 하루 단위 구간으로 나누어 줄이고(downsampling), `rolling("6h")`는 이동 창 통계를 내고, `tz_localize`·`tz_convert`는 시간대를 다룬다.

```python {exec="browser" session="sci"}
import pandas as pd

idx = pd.date_range("2024-07-01", periods=72, freq="h")   # 3일치 시간별 관측
hours = np.arange(72)                                    # 짝수 시각은 A, 홀수 시각은 B 관측소
obs = pd.DataFrame({
    "station": np.where(hours % 2 == 0, "A", "B"),
    "temp_c": 20 - 6 * (hours % 2) + 5 * np.sin(2 * np.pi * hours / 24),
}, index=idx)

daily = obs["temp_c"].resample("D").agg(["mean", "max"])
print(daily.round(2))

meta = pd.DataFrame({"station": ["A", "B"], "elev_m": [120, 850]})
by_station = obs.groupby("station")["temp_c"].mean().reset_index()
print(by_station.merge(meta, on="station", how="left").round(2))
```

```output
            mean   max
2024-07-01  17.0  25.0
2024-07-02  17.0  25.0
2024-07-03  17.0  25.0
  station  temp_c  elev_m
0       A    20.0     120
1       B    14.0     850
```

72개의 시간별 관측이 `resample`로 3개의 일별 행이 되었고, `groupby`로 관측소별 평균을 낸 뒤 `merge`로 고도 정보를 붙였다. 일별 평균 17.0은 고도가 다른 두 관측소를 섞은 값이다. 물리적으로 의미 있는 값을 원한다면 관측소별로 먼저 나눈 뒤 시간으로 묶어야 한다(`obs.groupby("station")["temp_c"].resample("D").mean()`). 무엇으로 먼저 묶을지는 코드가 아니라 질문이 정한다.

::quiz{id="resample-daily-rows"}

**pandas의 copy 규칙.** 집필 시점의 pandas(3.x)는 Copy-on-Write가 기본이다. 인덱싱으로 꺼낸 부분 DataFrame은 원본과 독립된 것처럼 동작하고, `df["b"][0] = 99` 같은 연쇄 할당(chained assignment)은 원본을 바꾸지 못한다(경고가 나온다). 원본의 일부를 바꾸려면 `df.loc[행, 열] = 값`처럼 한 번의 인덱싱으로 쓴다. 이전 버전의 pandas는 이 규칙이 달랐으므로, 오래된 코드나 자료의 예제를 옮길 때는 설치된 버전에서 동작을 확인한다.

**연구에서의 의미.** 시간 데이터에서 가장 흔한 오류는 시간대와 구간 경계다. UTC와 현지 시각이 섞인 두 표를 merge하면 몇 시간씩 어긋난 관측이 짝지어진다. resample의 구간이 왼쪽 닫힘인지, 결과의 시각 레이블이 구간의 시작인지 끝인지도 기본값을 확인해야 한다. 데이터의 의미와 단위를 정하는 일은 15.1장에서 다룬다.

## 대안 도구

pandas는 표준이지만 유일한 선택은 아니다. 데이터가 메모리 크기에 가깝거나 그보다 크고, 집계 쿼리가 대부분인 작업에서는 다음 도구들이 자주 쓰인다. 구체적인 API와 성능은 버전마다 빠르게 바뀌므로 공식 문서에서 확인한다.

- **Apache Arrow**: 열 지향(columnar) 메모리 형식의 표준이다. 서로 다른 라이브러리·언어가 같은 메모리 배치를 쓰면 복사와 변환 없이 데이터를 주고받을 수 있다. 저장 형식인 Parquet과 짝을 이룬다(15.2장).
- **Polars**: Rust로 작성된 DataFrame 라이브러리다. Arrow 메모리 형식을 쓰고, 여러 코어로 병렬 처리하며, 연산을 바로 실행하지 않고 쿼리 계획을 세워 최적화하는 lazy API를 제공한다.
- **DuckDB**: 프로세스 안에서 도는(in-process) 분석용 SQL 데이터베이스다. 별도 서버 없이 Parquet·CSV 파일이나 메모리의 DataFrame에 SQL을 바로 실행한다.

선택 기준은 단순하다. 메모리에 여유 있게 들어가는 표를 탐색하고 다른 과학 라이브러리와 섞어 쓴다면 pandas, 큰 파일을 반복해서 집계한다면 Polars나 DuckDB를 시험해 볼 가치가 있다. 어느 쪽이든 `to_numpy()`로 NumPy 배열을 얻어 수치 계산으로 넘어가는 경계는 같다.

## SciPy

SciPy는 NumPy 배열 위에서 동작하는 수치 알고리즘 모음이다. 직접 구현하기 쉬워 보이지만 수치적으로 까다로운 알고리즘(적응적 적분, 견고한 최적화, 희소 선형대수)을 검증된 구현으로 제공한다. 자주 쓰는 하위 모듈은 다음과 같다.

| 모듈 | 하는 일 | 이 책에서 다루는 곳 |
|---|---|---|
| `scipy.optimize` | 최소화, 근 찾기, 곡선 맞춤(`curve_fit`) | 9.4장, 10부 |
| `scipy.interpolate` | 1·다차원 보간, spline | 9.5장 |
| `scipy.integrate` | 수치 적분(`quad`), ODE 풀이(`solve_ivp`) | 9.5–9.6장 |
| `scipy.sparse` | 희소 행렬 저장과 연산, 희소 선형 시스템 | 5.4장, 9.3장 |
| `scipy.signal` | 필터, 스펙트럼, 합성곱 | 분야별 응용 |
| `scipy.stats` | 확률분포, 검정, 기술 통계 | 11–12부 |

```python {exec="browser" session="sci"}
from scipy import integrate, interpolate, optimize, sparse

t_s = np.linspace(0, 10, 11)                       # 관측 시각
v_ms = 9.81 * t_s                                  # 자유낙하 속도
v_of_t = interpolate.CubicSpline(t_s, v_ms)
dist_m, _ = integrate.quad(v_of_t, 0, 10)          # 속도를 적분한 거리
print("보간 v(2.5 s) =", round(float(v_of_t(2.5)), 4), "m/s,  거리 =", round(dist_m, 2), "m")

res = optimize.minimize_scalar(lambda x: (x - 2.0) ** 2 + 1.0)
print("최솟값 위치:", round(res.x, 6))

n = 1000
L = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="csr")  # 1차원 Laplacian
print("희소 저장 원소 수:", L.nnz, " 밀집이면:", n * n)
```

```output
보간 v(2.5 s) = 24.525 m/s,  거리 = 490.5 m
최솟값 위치: 2.0
희소 저장 원소 수: 2998  밀집이면: 1000000
```

1초 간격으로 기록한 속도를 spline으로 보간해 사이 시각의 값을 얻고, 보간 함수를 적분해 이동 거리 $\int_0^{10} 9.81\,t\,dt = 490.5$ m를 얻었다. 각 줄이 해석해와 맞는지 확인하는 습관이 중요하다. 1차원 Laplacian 행렬은 백만 칸 중 2998칸만 0이 아니다. 희소 형식은 0이 아닌 원소와 그 위치만 저장한다. 언제 희소 표현이 유리한지는 5.4장에서 다룬다.

**연구에서의 의미.** SciPy 함수는 결과와 함께 **수렴 정보**를 돌려준다. `optimize.minimize`의 `res.success`와 `res.message`, `integrate.quad`의 두 번째 반환값(오차 추정)을 버리지 말고 확인한다. 최적화가 반복 횟수 제한에 걸려 멈춘 결과를 최적해로 보고하는 일은 흔한 실수다.

:::note[분야 예시]
배터리 방전 실험에서는 전압·전류를 불규칙한 간격으로 기록하는 경우가 많다. 기록을 pandas로 읽어 시각 index로 정렬하고, `scipy.interpolate`로 공통 시간 격자에 맞춘 뒤, 전류를 `scipy.integrate`로 적분해 누적 전하량(Ah)을 구하고, `scipy.optimize.curve_fit`으로 등가회로 모델의 파라미터를 맞추는 것이 전형적인 흐름이다. 각 단계에서 단위(초와 시간, A와 mA)를 변수 이름에 붙여 두면 적분 결과가 3600배 어긋나는 실수를 막는다.
:::

## 실습

같은 계산을 Python 반복문과 두 가지 벡터화로 작성해 시간·메모리·수치 결과를 비교한다. 코드는 `labs/python-and-computing/scientific-python/loop_vs_vectorized.py`이고, 계산은 $N = 512$개 입자의 Lennard-Jones 퍼텐셜 에너지(무차원 단위)다.

$$
E = \sum_{i<j} 4\left(r_{ij}^{-12} - r_{ij}^{-6}\right)
$$

- `loop`: 이중 Python 반복문
- `broadcast`: `(N, N, D)` 차이 배열을 만드는 broadcasting
- `expand`: $\lVert\mathbf{x}\rVert^2 - 2\mathbf{x}^\top\mathbf{y} + \lVert\mathbf{y}\rVert^2$ 전개(행렬 곱 하나와 `(N, N)` 메모리)

이 기계에서의 예(시간은 3회 측정의 최솟값, 메모리는 `tracemalloc`으로 잰 최대 추가 할당량)는 다음과 같다. 시간 값은 기계마다 다르다.

```output
N = 512 입자, float64
구현                     에너지     시간(s)        추가 메모리(MB)
loop          -1958.886419    0.3691              0.00
broadcast     -1958.886419    0.0087             12.58
expand        -1958.886419    0.0030              6.29
broadcast 상대 오차(loop 기준): 1.7e-14,  속도 향상: 42배
expand 상대 오차(loop 기준): 1.7e-14,  속도 향상: 122배
float32 broadcast 상대 오차: 3.4e-08
float32 expand 상대 오차: 3.4e-07
```

결과를 세 축으로 읽는다.

- **시간.** 두 벡터화 모두 반복문보다 수십 배 이상 빠르고, 행렬 곱을 쓰는 `expand`가 가장 빠르다.
- **메모리.** `broadcast`는 `(N, N, D)` 중간 배열 때문에 `expand`의 약 두 배를 쓴다. $N$이 커지면 이 차이가 먼저 한계에 닿는다. $N$을 두 배로 늘리면 메모리가 몇 배가 되는지 예측하고 실행해 확인한다.
- **수치.** `float64`에서는 세 구현이 반올림 오차 수준($10^{-14}$)에서 일치한다. `float32`에서는 전개식 `expand`의 오차가 broadcasting보다 한 자릿수 크다. 비슷한 크기의 큰 수 $\lVert\mathbf{x}\rVert^2 + \lVert\mathbf{y}\rVert^2$에서 $2\mathbf{x}^\top\mathbf{y}$를 빼는 과정에서 유효 숫자가 사라지기 때문이다(소거 오차, 9.1장). 빠른 구현이 늘 같은 답을 주지는 않는다.

이어서 두 연습문제를 푼다. 하나는 쌍별 거리의 벡터화, 하나는 view·dtype·broadcasting 함정을 모두 피해야 하는 표준화다. 테스트가 바로 채점한다.

::exercise{id="pairwise-sq-dists"}

::exercise{id="standardize-columns"}

## 실패 사례

- **view를 수정해 원본이 바뀌는 것.** `sub = data[:, :10]; sub -= sub.mean(axis=0)`은 `data`를 바꾼다. 함수 안에서 입력의 slice를 제자리 연산으로 고치면, 호출한 쪽의 데이터가 조용히 바뀐다. 원본을 보존해야 하면 `.copy()`를 쓰거나 `sub = sub - sub.mean(axis=0)`처럼 새 배열을 만든다.
- **broadcasting으로 의도하지 않은 shape가 조용히 만들어지는 것.** `(N, 1)`과 `(N,)`의 연산은 `(N, N)`을 만든다. 손실·지표 계산 직전에 shape을 `assert`로 확인하고, 축을 줄일 때는 `keepdims=True`를 써서 차원을 유지한다.
- **정수 배열에 실수를 할당하는 것.** `out = np.zeros_like(labels)`처럼 정수 배열을 본떠 만든 출력 배열에 확률을 넣으면 모두 0이 된다. 출력 배열의 dtype을 명시한다.
- **`uint8` 이미지를 그대로 더하는 것.** 255를 넘는 값이 작은 값으로 돌아간다. 계산 전에 실수 dtype으로 바꾼다.
- **pandas에서 연쇄 할당으로 값을 바꾸려는 것.** `df[df.flag]["temp_c"] = np.nan`은 원본을 바꾸지 못한다. `df.loc[df.flag, "temp_c"] = np.nan`을 쓴다.

## 통과 기준

- shape 주석만 보고 복잡한 배열 연산(broadcasting, 축소, einsum이 섞인 5줄 안팎)의 결과 shape을 예측하고, 실행 결과와 일치한다.
- 반복문으로 쓴 계산 하나를 벡터화해 수치 결과가 일치함(`np.allclose`)을 확인하고, 시간과 메모리를 함께 측정해 보고한다.
- 연습문제 두 개(`pairwise-sq-dists`, `standardize-columns`)의 테스트를 통과한다.

## 참고 문헌

- NumPy 공식 문서 — *NumPy fundamentals*: Broadcasting, Copies and views, Data types. numpy.org.
- pandas 공식 문서 — *User Guide*: Group by, Merge/join, Time series, Copy-on-Write. pandas.pydata.org.
- SciPy 공식 문서 — *SciPy User Guide*. scipy.org.
