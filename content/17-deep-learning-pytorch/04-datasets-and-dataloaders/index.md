---
id: deep-learning-pytorch/datasets-and-dataloaders
title: "Dataset과 DataLoader"
title_en: "Datasets and DataLoaders"
tier: core
status: draft
prerequisites: [deep-learning-pytorch/tensors-and-autograd, data/splitting-and-leakage]
---

# Dataset과 DataLoader

모델 코드는 몇십 줄이지만, 과학 데이터를 모델이 먹을 수 있는 batch로 바꾸는 코드는 그보다 길고 버그도 많다. 관측소마다 길이가 다른 시계열, 원자 수가 다른 분자, 해상도가 다른 영상, 측정이 빠진 시각이 섞인 데이터를 같은 shape의 텐서로 묶어야 하기 때문이다. 그리고 이 단계의 버그는 대개 오류를 내지 않는다. padding된 0이 손실에 섞이거나, 여러 worker가 같은 난수로 증강하거나, 표준화 통계가 검증 데이터에서 계산되어도 학습은 돈다.

PyTorch는 이 일을 두 부분으로 나눈다. **Dataset**은 "인덱스 $i$의 sample 하나는 무엇인가"를 정의하고, **DataLoader**는 "어떤 순서로, 몇 개씩 묶어, 몇 개의 프로세스로 읽을 것인가"를 맡는다. 이 장은 둘 사이의 약속인 **sample contract**를 설계하는 것에서 시작해, 가변 길이 데이터의 batch 구성, sampler, 병렬 로딩의 동작을 다룬다. 데이터를 어떻게 나눌지(학습·검증·시험 분할과 누수 방지)는 15.4장의 원칙을 따른다고 가정한다.

## 학습 목표

- 데이터 계약(sample contract)을 설계한다
- 가변 길이 데이터를 batch로 만든다
- DataLoader의 병렬 동작을 이해한다

## Dataset

**정의.** PyTorch의 Dataset에는 두 종류가 있다.

- **map-style** (`torch.utils.data.Dataset`): `__len__`과 `__getitem__(i)`를 구현한다. 임의 접근이 가능하므로 shuffle·sampler를 자유롭게 쓸 수 있다. 메모리나 디스크에 있는 유한한 데이터의 기본 형태다.
- **iterable-style** (`torch.utils.data.IterableDataset`): `__iter__`만 구현한다. 순서대로 흘러나오는 데이터(스트리밍, 끝이 없는 시뮬레이션 출력, 원격 저장소의 샤드)에 쓴다. 인덱스가 없으므로 shuffle과 worker 간 분배를 Dataset이 직접 책임져야 한다(이 장의 대용량 데이터 절).

**`__getitem__`의 반환 계약.** `__getitem__(i)`는 sample **하나**를 돌려준다. batch를 만드는 것은 DataLoader의 collate 단계다. 반환값의 구조·shape·dtype이 모든 $i$에서 같은 규칙을 따라야 collate가 batch를 만들 수 있다. 같은 $i$에 대해 같은 결과를 내야 한다는 것도 계약의 일부다. 무작위 증강을 한다면 그 무작위성은 명시적으로 관리해야 한다(병렬 로딩 절).

```python {exec="python"}
import torch
from torch.utils.data import DataLoader, Dataset

class StationDataset(Dataset):
    """관측소별 기온 창(window) 하나를 sample 하나로 돌려준다."""
    def __init__(self, temps_k, window, station_ids):
        self.temps_k, self.window, self.ids = temps_k, window, station_ids   # (S, T_total) K

    def __len__(self):
        return self.temps_k.shape[0] * (self.temps_k.shape[1] - self.window)

    def __getitem__(self, i):
        s, t = divmod(i, self.temps_k.shape[1] - self.window)
        x = self.temps_k[s, t:t + self.window]             # (T,) 입력 창, K
        return {"x": x[:, None].float(),                   # (T, C=1) float32
                "y": self.temps_k[s, t + self.window].float(),   # () 다음 시각 기온, K
                "station": self.ids[s]}                    # metadata (문자열)

torch.manual_seed(0)
ds = StationDataset(280 + 5 * torch.randn(3, 50, dtype=torch.float64), window=24,
                    station_ids=["A", "B", "C"])
batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
print("len(ds) =", len(ds))
print({k: (tuple(v.shape), v.dtype) if torch.is_tensor(v) else v for k, v in batch.items()})
```

```output
len(ds) = 78
{'x': ((4, 24, 1), torch.float32), 'y': ((4,), torch.float32), 'station': ['B', 'A', 'A', 'B']}
```

관측소 3개 × 시작 시각 26개로 sample이 78개다. 원본은 float64였지만 `__getitem__`에서 float32로 바꿔 돌려주므로, 모델 쪽에서는 dtype을 신경 쓰지 않아도 된다(17.2장). DataLoader의 기본 collate는 dict의 같은 키끼리 모아, 텐서는 맨 앞에 batch 축을 붙여 쌓고 문자열은 list로 묶었다.

## sample contract

**왜 필요한가.** Dataset과 모델은 대개 다른 시점에, 때로는 다른 사람이 작성한다. 둘 사이에 "sample은 무엇을 담는가"에 대한 명시적 약속이 없으면, 모델 쪽 코드가 데이터의 의미를 추측하게 된다. 채널 순서, 단위, 결측의 표시 방법, 시간 축이 앞인지 뒤인지 같은 것들이다. 이 약속을 **sample contract**라 부르자.

**정의.** sample contract는 `__getitem__`이 돌려주는 각 필드의 이름·shape·dtype·단위·의미를 정한 명세다. 과학 데이터에서는 보통 네 종류의 필드가 필요하다.

| 필드 | 예 | 명시할 것 |
|---|---|---|
| 입력 | `x: (T, C) float32` | 축의 순서, 채널 이름과 단위, 표준화 여부 |
| 정답 | `y: () float32`, `labels: () int64` | 단위, 클래스 번호의 의미, 결측 표시 |
| mask | `mask: (T,) bool`, `y_valid: () bool` | `True`의 의미(관측 있음? padding 아님?) |
| metadata | `meta: {"station": str, "t0": str}` | 분할·그룹·사후 분석에 필요한 식별자 |

**직관.** sample contract는 함수의 타입 시그니처와 같다. 이름 있는 필드(dict나 dataclass)는 tuple보다 길지만, 필드를 추가해도 기존 코드가 깨지지 않고 순서를 틀릴 일이 없다. metadata는 학습에는 쓰지 않더라도, 나중에 "어느 관측소에서 오차가 큰가"를 분석하거나 그룹 단위 분할(15.4장)이 지켜졌는지 확인할 때 필요하다.

**연구에서의 의미.** contract는 문서가 아니라 **테스트**로 고정한다. "모든 sample의 `x`는 float32이고 `x.shape[1] == len(CHANNELS)`이며 `y`는 250–350 K 범위다" 같은 테스트는 데이터 버전이 바뀌거나 전처리를 고쳤을 때 조용한 오류를 잡는다. 실습의 `test_varlen_loader.py`가 그 예다.

::quiz{id="sample-contract"}

:::note[분야 예시]
단일세포 RNA 시퀀싱 데이터에서 sample 하나는 세포 하나이고, 입력은 유전자 수만큼의 발현량 벡터다. 같은 행렬이라도 원시 count인지, 라이브러리 크기로 정규화한 값인지, 로그 변환한 값인지에 따라 모델이 받는 분포가 완전히 다르다. sample contract에 "x는 log1p(CPM), float32, 유전자 순서는 genes.txt"처럼 변환과 순서를 적어 두지 않으면, 다른 전처리로 만든 데이터에 모델을 적용해도 오류 없이 엉뚱한 예측이 나온다. metadata에는 세포가 나온 기증자·배치(batch effect) 정보를 넣어 두어야 기증자 단위 분할을 할 수 있다.
:::

## batch 구성

**왜 필요한가.** 기본 collate는 텐서를 `torch.stack`으로 쌓으므로, 모든 sample의 shape이 같아야 한다. 과학 데이터는 길이가 다른 경우가 많다. 관측 기간이 다른 시계열, 원자 수가 다른 분자, 단어 수가 다른 문장이다. 해결책은 가장 긴 길이에 맞춰 뒤를 채우는 **padding**과, 어디가 실제 값인지 표시하는 **mask**다.

**padding과 mask.** 길이 $T_i$인 sample들을 $T_{\max} = \max_i T_i$로 채우면 batch는 $(B, T_{\max}, C)$이고, mask $m_{it} = [t < T_i]$는 $(B, T_{\max})$다. 손실은 mask가 참인 위치에서만 계산한다.

$$
L_{\text{원소}} = \frac{\sum_{i,t} m_{it}\,\ell_{it}}{\sum_{i,t} m_{it}}, \qquad
L_{\text{표본}} = \frac{1}{B}\sum_i \frac{\sum_t m_{it}\,\ell_{it}}{\sum_t m_{it}}
$$

$\ell_{it}$는 위치별 손실이다. $L_{\text{원소}}$는 모든 실제 시각에 같은 비중을 주므로 긴 시계열의 비중이 크고, $L_{\text{표본}}$은 모든 sample에 같은 비중을 준다. padding을 포함해 단순 평균하면 둘 다 아닌 값이 된다.

```python {exec="browser" session="dataloaders"}
import numpy as np

rng = np.random.default_rng(0)
lengths = [3, 7, 5]                                         # 가변 길이 시계열 3개
seqs = [rng.normal(loc=2.0, size=n) for n in lengths]
T = max(lengths)
x = np.zeros((len(seqs), T))                                # (B, T) 0으로 padding
mask = np.zeros((len(seqs), T), dtype=bool)                 # (B, T) 실제 값이면 True
for i, s in enumerate(seqs):
    x[i, :len(s)], mask[i, :len(s)] = s, True
pred = np.full_like(x, 2.0)                                 # 모든 시각에서 2.0을 예측
err = (pred - x) ** 2
print("padding 포함 MSE :", round(err.mean(), 4))
print("mask 적용 MSE    :", round(err[mask].mean(), 4))
print("표본별로 평균 후 평균:", round(np.mean([e[m].mean() for e, m in zip(err, mask)]), 4))
print("padding 비율     :", round(1 - mask.mean(), 3))
```

```output
padding 포함 MSE : 1.7601
mask 적용 MSE    : 0.8641
표본별로 평균 후 평균: 0.7863
padding 비율     : 0.286
```

padding 위치에서는 예측 2.0과 padding 값 0의 차이 $2^2 = 4$가 손실에 섞여, MSE가 두 배 넘게 부풀었다. padding 값이 예측과 우연히 같았다면 반대로 손실이 줄어들었을 것이다. 어느 쪽이든 padding 비율(여기서 28.6%)이 batch마다 다르므로, 손실의 의미가 batch 구성에 따라 바뀐다.

::quiz{id="padding-in-loss"}

::quiz{id="element-vs-sample-mean"}

**collate_fn.** DataLoader에 `collate_fn`을 넘기면 sample 목록을 batch로 만드는 방식을 바꿀 수 있다. padding·mask·길이를 함께 돌려주는 collate 함수는 다음과 같다.

```python {exec="python"}
from torch.nn.utils.rnn import pad_sequence

def make_seq(n, seed):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(n, 2, generator=g), "y": torch.tensor(float(n))}   # x: (T_i, C)

samples = [make_seq(n, i) for i, n in enumerate([3, 7, 5])]
try:
    torch.utils.data.default_collate(samples)
except RuntimeError as e:
    print("default_collate -> RuntimeError:", "equal size" in str(e))

def pad_collate(samples):
    xs = [s["x"] for s in samples]
    lengths = torch.tensor([len(x) for x in xs])                         # (B,)
    x = pad_sequence(xs, batch_first=True, padding_value=0.0)            # (B, T_max, C)
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]          # (B, T_max)
    return {"x": x, "mask": mask, "lengths": lengths, "y": torch.stack([s["y"] for s in samples])}

b = pad_collate(samples)
print({k: tuple(v.shape) for k, v in b.items()})
print(b["mask"].int())
```

```output
default_collate -> RuntimeError: True
{'x': (3, 7, 2), 'mask': (3, 7), 'lengths': (3,), 'y': (3,)}
tensor([[1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0]], dtype=torch.int32)
```

`DataLoader(ds, batch_size=..., collate_fn=pad_collate)`로 쓴다. mask는 손실뿐 아니라 **모델 안에서도** 필요하다. 시간 축 평균 pooling, attention, 정규화 층의 통계에 padding이 섞이면 같은 문제가 모델 내부에서 생긴다. 18장의 시계열·Transformer 모델은 mask를 입력으로 받는다.

::quiz{id="default-collate-varlen"}

**padding을 줄이는 방법.** batch 안의 길이 차이가 크면 계산의 대부분이 padding에 쓰인다. 길이가 비슷한 sample끼리 묶는 bucketing(길이로 정렬한 뒤 근처끼리 batch를 만들고, batch 순서만 섞는다)이 흔한 대응이다. 너무 긴 sample은 정해진 길이의 창으로 잘라 여러 sample로 만들 수도 있는데, 이때 같은 원본에서 나온 창들이 학습과 검증에 나뉘어 들어가지 않도록 분할은 원본 단위로 한다(15.4장).

## sampler

**정의.** sampler는 한 epoch 동안 Dataset의 어떤 인덱스를 어떤 순서로 방문할지 정한다. DataLoader의 `shuffle=True`는 무작위 순열을 만드는 sampler의 단축 표기다. 그 순열을 `batch_size`로 자르는 것이 batch sampler이고, 마지막 batch가 모자랄 때 버리는 옵션이 `drop_last`다.

- **shuffle**: epoch마다 다른 순열. DataLoader의 `generator` 인자에 시드를 고정한 `torch.Generator`를 넘기면 epoch별 순서가 재현된다.
- **weighted sampler**: 표본마다 가중치를 주고 그 비율로 복원 추출한다(집필 시점 예시: `torch.utils.data.WeightedRandomSampler`). 클래스가 불균형할 때 드문 클래스를 더 자주 보게 한다.
- **group 단위 sampler**: 같은 그룹(같은 환자, 같은 길이 bucket, 같은 시뮬레이션 run)의 표본을 한 batch에 모으거나, 반대로 한 batch에 한 그룹이 몰리지 않게 한다. `batch_sampler` 인자로 batch 인덱스 목록을 직접 정할 수 있다.

```python {exec="browser" session="dataloaders"}
rng = np.random.default_rng(0)
N, B = 10, 4
for epoch in range(2):
    perm = np.random.default_rng([0, epoch]).permutation(N)   # (시드, epoch)에서 결정
    batches = [perm[i:i + B].tolist() for i in range(0, N, B)]
    print(f"epoch {epoch}:", batches, " drop_last면 마지막 batch 버림")

labels = np.array([0] * 95 + [1] * 5)                       # 클래스 1이 5%
w = 1.0 / np.bincount(labels)[labels]                       # 표본 가중치 = 1 / 클래스 빈도
idx = rng.choice(len(labels), size=2000, replace=True, p=w / w.sum())
print("균일 추출 시 클래스 1 비율: 0.05,  가중 추출:", round(labels[idx].mean(), 3))
print("가중 추출에서 서로 다른 클래스 1 표본 수:", len(set(idx[labels[idx] == 1])))
```

```output
epoch 0: [[4, 6, 2, 7], [3, 5, 9, 0], [8, 1]]  drop_last면 마지막 batch 버림
epoch 1: [[9, 1, 3, 8], [7, 6, 0, 4], [2, 5]]  drop_last면 마지막 batch 버림
균일 추출 시 클래스 1 비율: 0.05,  가중 추출: 0.496
가중 추출에서 서로 다른 클래스 1 표본 수: 5
```

가중 추출은 클래스 비율을 0.05에서 0.5 근처로 맞췄지만, 클래스 1의 서로 다른 표본은 여전히 5개뿐이다. 같은 5개를 약 200번씩 반복해서 보는 것이다. 드문 클래스의 **정보**가 늘어나는 것이 아니므로 과적합 위험이 커지고, 학습 분포가 실제 분포와 달라져 예측 확률의 보정(calibration)이 깨진다. 가중 추출을 쓰면 평가는 원래 분포의 데이터로 하고, 확률을 그대로 쓸 일이 있으면 보정을 다시 확인한다(21.2장).

## 병렬 로딩

**왜 필요한가.** GPU가 batch 하나를 처리하는 동안 CPU는 다음 batch를 읽고 전처리해야 한다. 이 일을 학습 루프와 같은 프로세스에서 하면 GPU가 데이터를 기다리며 논다. DataLoader는 `num_workers`개의 **별도 프로세스**에서 `__getitem__`과 collate를 실행해 이 일을 겹친다.

**동작.** 집필 시점 PyTorch에서 map-style Dataset의 병렬 로딩은 다음과 같이 동작한다.

1. 메인 프로세스의 sampler가 batch 인덱스 목록을 만든다.
2. batch를 worker들에게 차례로 배분한다. worker $k$는 batch $k, k + W, k + 2W, \dots$를 맡는다.
3. 각 worker는 **Dataset의 복사본**으로 sample을 읽고 collate해 batch를 메인 프로세스에 보낸다. worker당 미리 만들어 두는 batch 수가 `prefetch_factor`다.
4. 메인 프로세스는 batch를 원래 순서대로 내놓는다(`in_order=True`가 기본값). 그래서 `num_workers`를 바꿔도 batch 순서는 같다.

관련 옵션은 다음과 같다.

| 옵션 | 의미 | 주의 |
|---|---|---|
| `num_workers` | worker 프로세스 수 | 0이면 메인 프로세스에서 읽는다. 디버깅은 0으로 |
| `pin_memory` | batch를 고정(page-locked) 메모리에 둔다 | CPU→GPU 복사를 빠르게, `.to(device, non_blocking=True)`와 함께 |
| `prefetch_factor` | worker당 미리 준비할 batch 수 | 메모리 사용량과 교환 |
| `persistent_workers` | epoch이 끝나도 worker를 유지 | worker 시작 비용을 줄인다. worker 안의 상태도 epoch을 넘어 유지된다 |
| `worker_init_fn` | worker 시작 시 호출하는 함수 | worker별 시드, 파일 핸들 열기 |

**worker별 seed.** worker는 Dataset의 복사본을 받는다(fork 방식에서는 프로세스 메모리가 복사되고, spawn·forkserver 방식에서는 Dataset이 pickle로 전달된다. 어느 방식이 기본인지는 운영체제와 Python 버전에 따라 다르다). 그래서 Dataset 안에 난수 생성기 객체가 있으면 **모든 worker가 같은 상태의 생성기**를 갖고 같은 난수열을 낸다. 집필 시점 PyTorch는 worker마다 Python `random`, `torch`, NumPy의 전역 난수 상태를 서로 다르게 시드하지만, 사용자가 만든 `np.random.Generator` 객체는 건드리지 않는다. worker가 받는 복사본을 pickle로 흉내 내 확인한다.

```python {exec="python"}
import pickle
import numpy as np

class NoisyDataset(Dataset):
    def __init__(self):
        self.rng = np.random.default_rng(0)                  # __init__에서 만든 Generator
    def __len__(self):
        return 100
    def __getitem__(self, i):
        return self.rng.normal()                             # 증강용 잡음이라고 하자

def worker_init_fn(worker_id):                               # DataLoader(worker_init_fn=...)
    info = torch.utils.data.get_worker_info()                # worker 안에서만 None이 아니다
    info.dataset.rng = np.random.default_rng(info.seed)      # info.seed는 worker마다 다르다

ds = NoisyDataset()
workers = [pickle.loads(pickle.dumps(ds)) for _ in range(2)]   # worker마다 복사본이 간다
print("복사본 그대로      :", [round(w[0], 4) for w in workers])
for worker_id, w in enumerate(workers):                      # worker_init_fn이 하는 일을 손으로
    w.rng = np.random.default_rng(1234 + worker_id)          # seed = base_seed + worker_id
print("worker별로 다시 시드:", [round(w[0], 4) for w in workers])
```

```output
복사본 그대로      : [0.1257, 0.1257]
worker별로 다시 시드: [-1.6038, -0.9097]
```

두 복사본이 같은 인덱스에 같은 잡음을 냈다. 실제 DataLoader에서는 서로 다른 표본에 같은 잡음 순서가 더해진다. `worker_init_fn`에서 `get_worker_info().seed`로 다시 시드하면 worker마다 다른 난수열을 쓴다. 이 seed는 DataLoader의 `generator`에서 epoch마다 새로 뽑으므로, `generator`를 고정하면 실행 전체가 재현된다. 실제 worker 프로세스로 같은 현상을 재현하는 코드는 실습에 있다.

::quiz{id="worker-same-random"}

**복사본이라는 사실의 다른 결과.** worker 안에서 Dataset 속성을 바꾸면(예: 읽은 파일을 dict에 캐시) 그 변경은 그 worker에만 남는다. 메인 프로세스나 다른 worker는 모른다. 또한 Dataset이 큰 Python 객체(list of dict 등)를 들고 있으면 fork 후 참조 횟수 갱신 때문에 메모리 페이지가 worker마다 복사되어, 메모리 사용량이 worker 수만큼 늘어나는 것처럼 보일 수 있다. 큰 데이터는 NumPy 배열이나 memory mapping(다음 절)으로 들고 있는 것이 안전하다.

::quiz{id="num-workers-facts"}

**몇 개의 worker가 적당한가.** 정답은 없다. GPU 사용률이 낮고 데이터 로딩이 병목이면 늘리고, CPU 코어 수와 메모리가 상한이다. 22.4장의 profiling으로 "GPU가 데이터를 기다리는 시간"을 직접 재서 정한다. 디버깅할 때는 `num_workers=0`으로 두어야 Dataset 안의 오류가 원래 traceback과 함께 보이고 디버거도 쓸 수 있다.

## 대용량 데이터

데이터가 메모리에 들어가지 않으면 Dataset이 디스크나 네트워크에서 필요한 부분만 읽어야 한다. 세 가지 방법이 있다.

- **memory mapping.** `np.load(path, mmap_mode="r")`나 `np.memmap`은 파일을 메모리처럼 indexing하되, 실제로 접근한 부분만 운영체제가 읽어 들인다. 모든 worker가 같은 파일을 매핑하므로 복사 문제도 없다. 고정 크기 배열(시계열 창, 이미지 패치)을 하나의 큰 이진 파일로 미리 변환해 두면 map-style Dataset을 거의 그대로 쓸 수 있다. 무작위 접근이 많으면 저장 장치의 임의 읽기 성능이 병목이 된다.
- **streaming.** 데이터를 여러 개의 샤드 파일로 나누고, iterable-style Dataset이 샤드를 순서대로 읽으며 sample을 흘려보낸다. 원격 저장소의 대규모 데이터에 쓴다. 완전한 무작위 shuffle은 불가능하므로, 샤드 순서를 섞고 일정 크기의 버퍼 안에서 섞는 근사를 쓴다. worker가 여러 개이면 `get_worker_info()`로 자기 몫의 샤드를 나눠 가져야 같은 sample이 worker 수만큼 중복되지 않는다.
- **캐싱.** 비싼 전처리(원시 파일 파싱, 특성 계산)의 결과를 처음 한 번만 계산해 디스크에 저장한다. 캐시 파일 이름에 전처리 설정의 해시를 넣어, 설정이 바뀌면 캐시도 새로 만들게 한다. 그렇지 않으면 전처리를 고쳤는데 옛 캐시로 학습하는 일이 생긴다.

:::note[분야 예시]
기후 재분석 자료처럼 수 TB에 이르는 격자 데이터는 시간·공간 조각(chunk)으로 나뉜 형식으로 저장하는 경우가 많다. Dataset이 "어느 시각·영역을 읽을지"만 정하고 해당 조각만 읽게 하면 전체를 메모리에 올리지 않고 학습할 수 있다. 이때 batch의 표본들이 같은 조각에 몰려 있으면 읽기가 빠르지만 batch 안의 상관이 커지고, 흩어져 있으면 그 반대다. 읽기 효율과 표본의 독립성 사이의 절충을 sampler 설계에서 정한다.
:::

## 실습

가변 길이 시계열을 padding·mask로 batch화하는 Dataset·collate_fn을 작성하고 테스트한다. 코드는 `labs/deep-learning-pytorch/datasets-and-dataloaders/varlen_loader.py`, 테스트는 같은 폴더의 `test_varlen_loader.py`다.

- `SensorSeqDataset`: 길이 20–100인 2채널 센서 시계열(온도 K, 압력 kPa) 64개. docstring에 sample contract(`x`, `y`, `length`, `meta`)를 적었다.
- `pad_collate`: `(B, T_max, C)` 입력, `(B, T_max)` mask, 길이, 정답, metadata 목록을 만든다.
- `masked_mean`: padding을 뺀 표본별 평균.
- 테스트는 sample contract(dtype·shape·단위 범위), 원본과 메모리를 공유하지 않는지, padding 위치와 값, padding 값을 바꿔도 masked 통계가 같은지, 한 epoch에 모든 표본을 한 번씩 방문하는지, worker별 난수가 다른지를 확인한다. 데이터 로더를 **모델 없이 단독으로** 테스트한다는 점이 핵심이다.
- batch 하나를 `outputs/batch.png`로 그려, 실제 값과 padding을 사람이 눈으로 확인한다.

집필 시점의 실행 결과는 다음과 같다(PyTorch 2.15, CPU, Linux).

```output
[1] batch 하나
  x        (4, 100, 2)
  mask     (4, 100)
  lengths  (4,)
  y        (4,)
  meta     [{'run_id': 'run-053'}, {'run_id': 'run-052'}, {'run_id': 'run-029'}, {'run_id': 'run-032'}]
  lengths: [100, 57, 79, 27]  mask 행 합: [100, 57, 79, 27]
  padding 포함 평균 온도(K): [299.4, 171.2, 237.0, 81.3]
  mask 적용 평균 온도(K)   : [299.4, 300.3, 300.0, 300.9]
  목표 y(K)               : [299.4, 300.3, 300.0, 300.9]
  그림 저장: outputs/batch.png
[2] 각 batch 첫 원소에 더해진 잡음 (batch 0·2는 worker 0, 1·3은 worker 1)
  worker_init_fn 없음 : [0.3456, 0.3456, 0.0541, 0.0541]
  reseed_worker 사용  : [-0.4068, 0.4672, -0.5381, 1.3467]
```

[1]에서 padding을 포함한 평균 온도는 길이 27인 시계열에서 81.3 K로, 물리적으로 불가능한 값이다. 단위가 있는 양은 이런 sanity check가 쉽다는 장점이 있다. [2]는 실제 worker 프로세스 2개로 읽은 결과다. `worker_init_fn`이 없으면 worker 0과 1이 같은 잡음(0.3456, 0.0541)을 차례로 냈고, 다시 시드하면 모두 다르다.

이어서 두 연습문제를 푼다. 첫째는 padding·mask와 masked 손실을 NumPy로 구현하는 것이고, 둘째는 sampler와 worker 배분의 동작을 구현하는 것이다.

::exercise{id="pad-and-mask"}

::exercise{id="batch-indices"}

## 실패 사례

- **worker들이 같은 random seed를 공유하는 것.** Dataset 안에 만든 Generator 객체는 worker마다 복사되어 같은 난수열을 낸다. 증강이 worker 수만큼 중복되고 오류는 나지 않는다. `worker_init_fn`에서 `get_worker_info().seed`로 다시 시드하고, 시드를 바꿔 가며 batch를 몇 개 뽑아 실제로 다른지 테스트한다.
- **padding된 값이 손실과 metric에 섞이는 것.** padding 0이 "완벽히 맞힌 예측"이나 "0 K"가 되어 손실·metric을 왜곡한다. 손실, metric, 모델 내부의 pooling·attention·정규화에 모두 mask를 적용하고, padding 값을 바꿔도 결과가 같은지 테스트한다.
- **표준화 통계를 전체 데이터에서 계산하는 것.** Dataset 안에서 전체 데이터의 평균·표준편차로 표준화하면 검증·시험 데이터의 정보가 학습에 새어 든다. 통계는 학습 분할에서만 계산해 Dataset에 넘긴다(15.4장).
- **병렬 로딩 상태에서 디버깅하는 것.** worker 안의 오류는 다른 프로세스에서 나므로 traceback이 잘리거나 디버거가 멈추지 않는다. 문제가 생기면 `num_workers=0`으로 재현한다.

## 통과 기준

- 실습의 Dataset·collate_fn을 모델 없이 단독으로 테스트하고(sample contract, padding 위치와 값, masked 통계의 불변성, epoch 커버리지, worker별 난수), batch 하나를 그림으로 그려 사람이 검증한다.
- 자기 데이터의 sample contract를 필드별 이름·shape·dtype·단위·의미의 표로 쓰고, 그 표를 테스트로 옮긴다.
- 연습문제 두 개(`pad-and-mask`, `batch-indices`)의 테스트를 통과한다.
