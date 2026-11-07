---
id: data/storage-formats
title: "저장 형식과 대용량 데이터"
title_en: "Storage Formats and Large Data"
tier: core
status: draft
prerequisites: [data/data-semantics]
---

# 저장 형식과 대용량 데이터

연구 초기에는 데이터가 작아서 어떤 형식으로 저장하든 문제가 없다. CSV 하나를 열어 pandas로 읽으면 끝난다. 그러다 측정이 쌓이고 시뮬레이션 출력이 수십 GB가 되면, 실험 한 번에 데이터를 읽는 데만 몇 분이 걸리고, 메모리가 부족해 프로세스가 죽고, 공유 서버의 파일시스템이 느려진다. 이때 모델을 바꾸는 것으로는 해결되지 않는다. 데이터를 **어떤 형식으로, 어떤 단위로 나누어, 어떤 순서로** 저장하고 읽는지가 연구 속도를 정한다.

이 장은 과학 데이터에서 자주 만나는 저장 형식을 비교하고, 열 지향 저장과 압축이 왜 빠른지, 메모리보다 큰 데이터를 어떻게 처리하는지, 학습 시의 접근 패턴이 형식 선택을 어떻게 바꾸는지를 다룬다. 브라우저에서 실행할 수 있는 CSV·JSON·NPY는 직접 실행하고, 별도 라이브러리가 필요한 Parquet·HDF5·Zarr·DuckDB·Polars는 개념과 "집필 시점 예시" 코드로 설명한다.

## 학습 목표

- 데이터 형식별 장단점을 안다
- 메모리보다 큰 데이터를 처리한다

## 형식 비교

**왜 필요한가.** 저장 형식은 한 번 정하면 바꾸기 어렵다. 데이터를 만드는 장비·시뮬레이터와 읽는 코드가 모두 그 형식에 맞춰지기 때문이다. 형식마다 잘하는 일이 다르다. 사람이 열어 볼 수 있는가, 값을 정확히 보존하는가, 일부만 읽을 수 있는가, 여러 차원의 배열을 담는가, metadata를 함께 담는가를 기준으로 고른다.

**주요 형식.**

| 형식 | 구조 | 장점 | 단점 | 어울리는 데이터 |
|---|---|---|---|---|
| CSV | 행 단위 텍스트 표 | 어디서나 열린다. 사람이 읽는다 | 느린 파싱, 자료형·단위 정보 없음, 반올림 위험 | 작은 표, 데이터 교환 |
| JSON Lines | 한 줄에 JSON 객체 하나 | 행마다 구조가 달라도 된다 | 키 이름 반복으로 크다 | 로그, 메타데이터, 불규칙한 레코드 |
| NPY / NPZ | NumPy 배열의 이진 저장 | 정확한 값, 빠른 읽기, memory mapping | NumPy 밖에서 쓰기 불편, 열 이름 없음 | 단일 배열, 중간 결과 |
| MAT | MATLAB 파일 | MATLAB 사용 그룹과 교환 | v7.3은 내부가 HDF5라 별도 도구 필요 | MATLAB 기반 실험 데이터 |
| Parquet | 열 지향 이진 표 | 열 단위 읽기, 압축, 자료형 보존, 분할 저장 | 행 하나만 자주 고치는 용도에 부적합 | 큰 표, 분석 질의 |
| Arrow (IPC/Feather) | 열 지향 메모리 형식 | 복사 없이 프로세스·언어 간 공유 | 장기 보관용 압축은 Parquet보다 약함 | 도구 사이의 빠른 교환 |
| HDF5 | 계층(그룹·데이터셋) 이진 파일 | 다차원 배열, chunk 단위 부분 읽기, 속성(attrs)에 metadata | 단일 파일의 동시 쓰기가 어렵다 | 시뮬레이션 출력, 실험 장비 데이터 |
| Zarr | chunk마다 별도 객체로 저장하는 배열 | 클라우드 객체 저장소와 병렬 쓰기에 적합 | 작은 chunk가 많으면 작은 파일 문제 | 기후·영상 같은 거대 다차원 배열 |

**정확성.** 텍스트 형식은 숫자를 글자로 바꾸어 저장한다. 쓰는 자릿수가 부족하면 그만큼 정보가 사라진다. 이진 형식은 메모리의 비트를 그대로 저장하므로 다시 읽으면 정확히 같은 값이 나온다. 같은 데이터(10 Hz로 기록한 타임스탬프·전압·온도 1만 행)를 세 형식으로 저장해 크기와 정확성을 비교한다.

```python {exec="browser" session="formats"}
import io
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 10_000
t_ms = np.arange(n, dtype=np.int64) * 100                     # (10000,) 10 Hz 타임스탬프
df = pd.DataFrame({
    "t_ms": t_ms,
    "voltage_v": 3.7 + 0.01 * np.sin(t_ms / 1000) + rng.normal(0, 1e-3, n),
    "temp_c": 25.0 + 2e-6 * t_ms + rng.normal(0, 0.05, n),
})                                                            # (10000, 3)

csv_bytes = df.to_csv(index=False).encode()
jsonl_bytes = "\n".join(json.dumps(r) for r in df.to_dict("records")).encode()
buf = io.BytesIO()
np.save(buf, df.to_numpy(dtype=np.float64))                   # (10000, 3) float64
npy_bytes = buf.getvalue()
for name, b in [("CSV", csv_bytes), ("JSONL", jsonl_bytes), ("NPY", npy_bytes)]:
    print(f"{name:5s} {len(b):>9,d} bytes")

back = np.loadtxt(io.BytesIO(csv_bytes), delimiter=",", skiprows=1)   # (10000, 3)
print("CSV 왕복 후 값이 같은가:", np.array_equal(back, df.to_numpy(dtype=np.float64)))
short = pd.read_csv(io.StringIO(df.to_csv(index=False, float_format="%.3f")))
print(f"소수 3자리 CSV의 최대 오차: {np.abs(short['voltage_v'] - df['voltage_v']).max():.1e} V")
```

```output
CSV     440,170 bytes
JSONL   790,147 bytes
NPY     240,128 bytes
CSV 왕복 후 값이 같은가: True
소수 3자리 CSV의 최대 오차: 5.0e-04 V
```

세 가지를 읽을 수 있다.

- **크기.** 이진 NPY는 값 하나에 정확히 8 bytes를 쓴다(3만 개 × 8 = 240,000에 헤더 128 bytes). CSV는 숫자를 십진 글자로 쓰므로 거의 두 배이고, JSON Lines는 줄마다 키 이름을 반복해 세 배가 넘는다.
- **왕복.** pandas는 float를 다시 읽었을 때 같은 값이 되는 최소 자릿수로 쓰고, 여기서는 Python의 float 변환을 쓰는 `np.loadtxt`로 다시 읽었으므로 왕복이 정확했다. 그러나 이것은 쓰는 쪽과 읽는 쪽 도구의 설정에 기댄 결과다. 예를 들어 pandas의 `read_csv`가 기본으로 쓰는 float 변환기는 버전에 따라 마지막 비트가 달라질 수 있다. 정확한 왕복이 중요하면 직접 확인한다.
- **반올림.** `float_format="%.3f"`처럼 자릿수를 줄이면 최대 0.0005 V의 오차가 생긴다. 센서 분해능이 0.001 V보다 좋다면 저장하면서 측정 정밀도를 버린 셈이다.

::quiz{id="csv-rounding-error"}

**metadata를 함께 담는가.** CSV에는 단위·자료형·결측 규칙을 넣을 자리가 없다. 15.1장의 데이터 사전을 별도 파일로 두어야 한다. Parquet은 열마다 자료형을 저장하고 파일 수준의 key-value metadata를 담을 수 있다. HDF5는 데이터셋마다 속성(attribute)을 달 수 있어 단위·보정 계수를 값 옆에 둘 수 있다. 아래는 HDF5에 chunk와 압축, 단위 속성을 지정해 저장하고 일부만 읽는 예다(집필 시점 예시, `h5py` 필요).

```python
import h5py
import numpy as np

voltage_v = np.random.default_rng(0).normal(3.7, 0.01, size=10_000_000)   # (10000000,)
with h5py.File("run.h5", "w") as f:
    dset = f.create_dataset("voltage_v", data=voltage_v,
                            chunks=(65_536,), compression="gzip")
    dset.attrs["unit"] = "V"
    dset.attrs["sampling_hz"] = 1000

with h5py.File("run.h5", "r") as f:
    part = f["voltage_v"][1_000_000:1_010_000]     # 이 구간을 포함한 chunk만 읽는다
    print(part.shape, f["voltage_v"].attrs["unit"])
```

:::note[분야 예시]
기후·해양 모델의 출력은 (시간, 깊이, 위도, 경도) 같은 4차원 격자다. 오랫동안 NetCDF(내부적으로 HDF5를 쓰는 버전이 있다)가 표준이었고, 클라우드에 올린 거대 격자에는 chunk를 객체 하나씩으로 저장하는 Zarr가 많이 쓰인다. 한 지점의 긴 시계열을 자주 읽는지, 한 시각의 전 지구 지도를 자주 읽는지에 따라 chunk 모양을 다르게 잡는다. 생물정보학의 단일세포 데이터처럼 큰 희소 행렬과 주석 표를 함께 담는 분야에서도 HDF5 기반 형식이 널리 쓰인다.
:::

## 열 지향 저장과 압축

**왜 필요한가.** 분석 질의는 대개 "모든 행의 몇몇 열"을 읽는다. 열 200개 중 전압과 온도의 평균만 필요하다면 나머지 198개 열은 읽을 필요가 없다. 행 단위 형식은 한 행의 값들이 붙어 있어 필요한 열만 골라 읽을 수 없다.

**정의.** **행 지향**(row-oriented) 저장은 한 행의 값들을 이어 붙이고 다음 행으로 넘어간다. **열 지향**(column-oriented) 저장은 한 열의 값들을 모두 이어 붙이고 다음 열로 넘어간다. Parquet은 행을 수십만~수백만 개씩 묶은 **row group** 안에서 열마다 따로 저장하고, 열마다 최솟값·최댓값 같은 통계를 기록한다. 그래서 필요한 열만 읽고(projection), 조건에 맞을 수 없는 row group은 건너뛴다(predicate pushdown).

**압축과의 관계.** 같은 열의 값은 자료형과 크기가 비슷하고, 시간 순서로 기록된 값은 이웃끼리 비슷하다. 그래서 열 단위로 모으면 압축 알고리즘이 반복 패턴을 더 잘 찾는다. 타임스탬프처럼 일정하게 증가하는 열은 이웃한 값의 **차분**(delta encoding)만 저장하면 거의 같은 수의 반복이 된다.

```python {exec="browser" session="formats"}
import zlib

X = np.ascontiguousarray(df.to_numpy(dtype=np.float64))       # (10000, 3), 행 우선(C) 배치
row_major = X.tobytes(order="C")                              # 행 단위: t, v, T, t, v, T, ...
col_major = X.tobytes(order="F")                              # 열 단위: t, t, ..., v, v, ..., T, T, ...
t_delta = np.diff(t_ms, prepend=0).tobytes()                  # 타임스탬프의 차분: 거의 모두 100
for name, b in [("행 지향", row_major), ("열 지향", col_major)]:
    print(f"{name}: {len(b) / 1000:.1f} kB → {len(zlib.compress(b, 6)) / 1000:.1f} kB")
t_raw = len(zlib.compress(t_ms.tobytes(), 6))
print(f"t_ms 열만: 그대로 {t_raw / 1000:.1f} kB, 차분 후 {len(zlib.compress(t_delta, 6)) / 1000:.1f} kB")
```

```output
행 지향: 240.0 kB → 174.2 kB
열 지향: 240.0 kB → 159.2 kB
t_ms 열만: 그대로 18.0 kB, 차분 후 0.1 kB
```

같은 바이트를 열 순서로 늘어놓기만 해도 압축 후 크기가 줄었다. 타임스탬프 열은 차분을 취하자 거의 모든 값이 100이 되어 18 kB가 0.1 kB 남짓으로 줄었다. (압축 결과의 정확한 byte 수는 zlib 버전에 따라 한두 byte 달라질 수 있어 kB 단위로 표시했다.) 전압·온도처럼 잡음이 섞인 float64 값은 하위 비트가 무작위라 범용 압축으로는 크게 줄지 않는다. 그래서 과학 데이터에서는 측정 분해능에 맞게 정수로 양자화하거나 float32로 저장하는 결정이 압축률을 크게 바꾼다. 이 결정은 정밀도를 버리는 일이므로 데이터 사전에 기록한다.

::quiz{id="columnar-partial-read"}

**압축 코덱.** Parquet은 열마다 압축 코덱을 고를 수 있다. 흔히 쓰이는 Snappy는 빠르지만 압축률이 낮고, Zstandard(zstd)와 gzip은 더 작게 만들지만 더 느리다. 보관용인지, 반복해서 읽는 학습용인지에 따라 고른다.

**partitioning.** 큰 표를 열 값에 따라 디렉터리로 나누어 저장하는 방식이다. 예를 들어 `cell_id=C03/part-0.parquet`처럼 셀마다 폴더를 두면 "셀 C03만" 읽는 질의는 다른 폴더를 열어 보지도 않는다. 다만 값의 종류가 너무 많은 열(예: 타임스탬프)로 나누면 작은 파일이 수없이 생긴다. 질의에서 자주 거르는, 값의 종류가 수십~수천 개인 열로 나눈다(집필 시점 예시, `pyarrow` 필요).

```python
import pandas as pd

df.to_parquet("cycles/", partition_cols=["cell_id"], compression="zstd")
part = pd.read_parquet("cycles/", columns=["cycle", "capacity_ah"],
                       filters=[("cell_id", "==", "C03")])
```

## 대용량 처리

**왜 필요한가.** 데이터가 RAM보다 크면 `pd.read_csv`로 통째로 읽을 수 없다. 더 큰 서버를 빌리는 것보다, 필요한 부분만 필요한 순간에 읽는 방식으로 바꾸는 것이 대개 더 싸고 빠르다.

**세 가지 기법.**

- **chunking.** 데이터를 덩어리로 나누어 하나씩 읽고, 통계량을 갱신하고, 버린다. 메모리 사용량은 덩어리 크기로 정해진다.
- **memory mapping.** 파일을 가상 메모리 주소에 대응시키고, 배열의 일부에 접근하는 순간 운영체제가 그 부분만 디스크에서 읽는다. `np.load(path, mmap_mode="r")`나 `np.memmap`이 이 방식이다. 고정 폭 이진 형식에서만 쓸 수 있다.
- **lazy evaluation.** 질의를 바로 실행하지 않고 계획으로 모아 두었다가, 마지막에 필요한 열과 행만 읽도록 최적화해 한 번에 실행한다.

chunking으로 CSV 전체의 평균을 구한다. 한 번에 2000행만 메모리에 있다.

```python {exec="browser" session="formats"}
import os
import tempfile

tmp = tempfile.mkdtemp()
csv_path = os.path.join(tmp, "series.csv")
df.to_csv(csv_path, index=False)

total, count, peak_rows = 0.0, 0, 0
for chunk in pd.read_csv(csv_path, chunksize=2_000):          # 한 번에 2000행만 메모리에
    total += chunk["temp_c"].sum()
    count += len(chunk)
    peak_rows = max(peak_rows, len(chunk))
print("행 수:", count, " 한 번에 읽은 최대 행 수:", peak_rows)
print("chunk 평균과 전체 평균의 차이 < 1e-9:", abs(total / count - df["temp_c"].mean()) < 1e-9)
```

```output
행 수: 10000  한 번에 읽은 최대 행 수: 2000
chunk 평균과 전체 평균의 차이 < 1e-9: True
```

평균은 합과 개수만 들고 다니면 되므로 덩어리로 나누어도 정확히 같다(합의 순서가 달라 생기는 1e-9 미만의 반올림 차이만 있다). 분산·분위수처럼 더 복잡한 통계량은 덩어리끼리 합치는 방법을 따로 설계해야 한다. 분산은 덩어리마다 개수·평균·편차 제곱합을 구해 병합하면 정확히 구할 수 있다. $\mathbb{E}[x^2] - (\mathbb{E}[x])^2$를 누적하는 방식은 큰 offset 위의 값에서 소거 오차로 무너진다(9.1장). 연습문제 `chunked-mean-var`에서 직접 구현한다.

::quiz{id="streaming-variance"}

**out-of-core 질의 도구.** 표 데이터에서는 chunk 루프를 직접 쓰기보다 질의 엔진에 맡기는 편이 낫다. DuckDB는 프로세스 안에서 도는 분석용 SQL 데이터베이스로, Parquet·CSV 파일을 바로 질의하고 메모리보다 큰 중간 결과는 디스크로 내려 처리할 수 있다. Polars는 lazy API로 질의 계획을 최적화해, 필요한 열과 row group만 읽는다. 아래 두 예는 같은 질의다(집필 시점 예시, `duckdb`·`polars` 필요. API 이름은 버전마다 바뀌므로 공식 문서에서 확인한다).

```python
import duckdb
import polars as pl

# SQL로: Parquet 파일 여러 개를 하나의 표처럼 질의
mean_by_cell = duckdb.sql("""
    SELECT cell_id, avg(voltage_v) AS mean_v
    FROM 'cycles/*/*.parquet'
    WHERE cycle > 100
    GROUP BY cell_id
""").df()

# lazy API로: collect() 전까지는 계획만 만든다
mean_by_cell = (pl.scan_parquet("cycles/*/*.parquet")
                  .filter(pl.col("cycle") > 100)
                  .group_by("cell_id")
                  .agg(pl.col("voltage_v").mean())
                  .collect())
```

:::tip[연구 팁]
수십 GB의 CSV를 실험마다 파싱하고 있다면, 한 번만 Parquet(표)이나 NPY·HDF5(배열)로 변환해 `data/interim/`에 둔다(15.1장). 변환 스크립트와 원본 checksum을 함께 기록하면(15.5장) 변환본을 지워도 다시 만들 수 있다.
:::

## 데이터 접근 패턴

**왜 필요한가.** 같은 형식이라도 읽는 순서에 따라 속도가 크게 다르다. 분석은 대개 큰 덩어리를 순서대로 읽지만(순차 접근), 학습은 매 epoch 샘플을 무작위 순서로 섞어 읽는다(random access). 저장장치는 순차 읽기에 최적화되어 있으므로, 학습용 데이터는 random access를 싸게 만드는 구조가 필요하다.

**고정 폭과 random access.** 이진 배열은 행마다 크기가 같으므로 $i$번째 행의 위치가 $\text{header} + i \times \text{행 크기}$로 계산된다. 앞의 데이터를 읽지 않고 그 위치로 바로 갈 수 있다. CSV는 줄마다 글자 수가 달라 $i$번째 줄을 찾으려면 앞의 줄을 모두 읽어야 한다.

```python {exec="browser" session="formats"}
npy_path = os.path.join(tmp, "series.npy")
np.save(npy_path, X)
header = os.path.getsize(npy_path) - X.nbytes                 # NPY 헤더 크기
i, row_bytes = 7_321, X.shape[1] * 8                          # 한 행 = 3 × 8 bytes
with open(npy_path, "rb") as f:
    f.seek(header + i * row_bytes)                            # 앞의 행을 읽지 않고 건너뜀
    row = np.frombuffer(f.read(row_bytes), dtype="<f8")       # (3,)
print("헤더", header, "bytes, 읽은 행이 원본과 같은가:", np.array_equal(row, X[i]))

line_len = [len(s) for s in open(csv_path).read().splitlines()[1:]]
print("CSV 한 줄의 길이:", min(line_len), "~", max(line_len), "글자 → 위치를 계산할 수 없다")
```

```output
헤더 128 bytes, 읽은 행이 원본과 같은가: True
CSV 한 줄의 길이: 38 ~ 44 글자 → 위치를 계산할 수 없다
```

::quiz{id="npy-row-offset"}

**작은 파일 문제.** 샘플마다 파일 하나(이미지 한 장, 시뮬레이션 한 번)를 두는 구조는 처음에는 편하다. 그러나 파일이 수백만 개가 되면 파일을 여는 비용, 디렉터리를 나열하는 비용, 공유 파일시스템의 메타데이터 서버에 보내는 요청이 데이터를 읽는 비용보다 커진다. 클러스터의 병렬 파일시스템은 큰 파일의 순차 읽기에 맞춰 설계되어 있어, 작은 파일 수백만 개는 자기 작업뿐 아니라 다른 사용자의 작업까지 느리게 만든다(2.5장).

해결책은 샘플을 수천 개씩 묶은 **shard** 파일로 저장하는 것이다. shard 안에서는 순차로 읽고, 학습에 필요한 무작위성은 shard 순서를 섞고 각 shard에서 읽은 샘플을 메모리의 작은 버퍼 안에서 다시 섞어 얻는다. 완전한 무작위 순서는 아니지만, 대개 학습에 충분한 수준의 섞임을 훨씬 싼 비용으로 얻는다.

::quiz{id="small-files"}

**선택의 기준.** 접근 패턴을 먼저 적고, 형식은 그 다음에 고른다.

| 주된 접근 | 권장 구조 |
|---|---|
| 열 몇 개로 집계·필터 | Parquet + 자주 거르는 열로 partitioning, DuckDB·Polars로 질의 |
| 다차원 배열의 일부 구간 | HDF5·Zarr, 읽는 모양에 맞춘 chunk |
| 학습 중 샘플 단위 random access, 데이터가 RAM보다 큼 | 고정 폭 이진 배열 + memory mapping, 또는 shard + 색인 |
| 작은 표, 교환, 사람이 확인 | CSV + 데이터 사전 |

## 실습

같은 시계열 데이터(예: 1 kHz로 기록한 전압·전류·온도 1억 행, 또는 자신의 연구 데이터)를 CSV·Parquet·HDF5로 저장하고 세 가지를 측정한다.

1. 파일 크기(압축 코덱별)
2. 전체를 읽는 시간
3. 부분 읽기 시간: (a) 열 하나 전체, (b) 시간 구간 하나의 모든 열

측정은 같은 파일을 여러 번 읽어 운영체제의 파일 캐시 효과를 확인하고, 첫 읽기와 반복 읽기를 구분해 기록한다. 결과를 표로 정리하고, 자신의 연구에서 가장 흔한 접근이 (a)와 (b) 중 무엇인지에 따라 형식을 고른 근거를 쓴다.

메모리보다 큰 데이터를 덩어리로 처리하는 핵심 계산을 연습문제로 구현한다.

::exercise{id="chunked-mean-var"}

## 실패 사례

- **수십 GB CSV를 매 실험마다 파싱하는 것.** 실험 시간의 대부분이 텍스트 파싱에 쓰인다. 한 번 이진 형식으로 변환해 두고, 변환 스크립트를 기록한다.
- **수백만 개의 작은 파일.** 공유 파일시스템에서 학습이 데이터를 기다리고, 다른 사용자의 작업까지 느려진다. shard로 묶는다.
- **저장하면서 정밀도를 버리는 것.** CSV의 자릿수, float32 변환, 정수 양자화는 모두 정보 손실이다. 손실의 크기를 측정 분해능과 비교하고 데이터 사전에 기록한다.
- **분산을 $\mathbb{E}[x^2] - (\mathbb{E}[x])^2$로 누적하는 것.** 큰 offset을 가진 값에서 소거 오차로 분산이 0이나 음수가 된다.
- **chunk 모양을 접근 패턴과 반대로 잡는 것.** 지점별 시계열을 자주 읽는데 시각별 지도 단위로 chunk를 잡으면, 시계열 하나를 읽으려고 모든 chunk를 연다.

## 통과 기준

- 자신의 데이터에서 CSV·Parquet·HDF5의 크기와 전체·부분 읽기 시간을 측정한 표를 만들고, 주된 접근 패턴에 근거해 형식과 chunk·partition 방식을 선택한다.
- 메모리보다 큰 데이터의 통계량을 덩어리 단위로 정확히 계산하는 방법을 설명한다.
- 연습문제 `chunked-mean-var`의 테스트를 통과한다.
