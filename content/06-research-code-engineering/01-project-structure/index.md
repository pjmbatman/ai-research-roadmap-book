---
id: research-code-engineering/project-structure
title: "연구 프로젝트 구조"
title_en: "Research Project Structure"
tier: core
status: draft
prerequisites: [python-and-computing/modules-and-imports]
---

# 연구 프로젝트 구조

연구 코드는 대개 notebook 하나나 `train.py` 하나에서 시작한다. 처음 몇 주는 그것으로 충분하다. 문제는 몇 달 뒤에 온다. 논문 표의 숫자가 어느 코드의 어느 버전에서 나왔는지 모르고, 평가 방식을 하나 바꾸려면 학습부터 다시 돌려야 하고, 동료는 어디부터 읽어야 할지 모른다. 이런 문제의 대부분은 코드를 잘못 짜서가 아니라 코드를 **어디에 두었는지** 정하지 않아서 생긴다.

이 장은 연구 프로젝트의 코드를 책임에 따라 나누고, 표준 디렉터리 구조에 배치하고, 학습·평가·분석을 서로 독립적으로 다시 실행할 수 있게 만드는 방법을 다룬다. 목표는 단순하다. 학습을 다시 하지 않고도 평가 스크립트만 실행해 같은 숫자를 얻을 수 있는 구조다.

## 학습 목표

- 책임에 따라 디렉터리를 나눈다
- notebook과 source code의 역할을 구분한다
- 연구실 공통 템플릿을 사용한다

## 책임 분리

**왜 필요한가.** 모든 코드가 한 파일에 있으면 한 가지를 바꿀 때 나머지가 함께 흔들린다. loss를 바꾸는 ablation을 하려다 데이터 전처리를 건드리고, 그림의 축 이름을 고치려다 학습 loop를 다시 실행한다. 바뀌는 이유가 다른 코드를 한곳에 두면, 바꾸지 않아도 될 코드까지 다시 실행하고 다시 검증해야 한다.

**정의.** 여기서 **책임**(responsibility)은 "이 코드가 바뀌는 이유 하나"를 뜻한다. 바뀌는 이유가 같은 코드는 한 모듈에, 다른 코드는 다른 모듈에 둔다. 연구 코드에서 자주 나뉘는 책임은 여섯 가지다.

| 책임 | 입력 | 출력 | 주로 바뀌는 이유 |
|---|---|---|---|
| 데이터 처리 | 원시 데이터 | 정제·분할된 데이터 | 결측치 처리, 분할 방식, 새 데이터셋 |
| 모델 | 설정(폭·깊이 등) | 모델 객체 | 구조 변경, 새 아키텍처 |
| loss | 예측, 정답 | 스칼라 손실 | 가중치, 정규화항, 물리 제약 |
| 학습 | 데이터, 모델, loss, 설정 | checkpoint, 학습 로그 | optimizer, scheduler, 학습 전략 |
| 평가 | checkpoint, 데이터 | metric, 예측값 | metric 정의, 평가 데이터 |
| 시각화 | metric, 예측값 | 그림, 표 | 논문 형식, 비교 대상 |

**직관.** 책임 사이의 경계는 **데이터가 넘어가는 지점**이다. 각 단계가 앞 단계의 저장된 산출물만으로 시작할 수 있으면 경계가 제대로 그어진 것이다. 시각화는 저장된 metric과 예측값만 있으면 되고, 평가는 저장된 checkpoint와 데이터만 있으면 된다. 경계를 정할 때 묻는 질문은 하나다. "이 부분만 바꿔서 다시 실행할 일이 있는가?" 그렇다면 그 부분은 앞뒤와 분리되어야 한다.

**연구에서의 의미.** 연구의 일상은 같은 파이프라인에서 한 부분만 바꾸는 실험의 반복이다. loss만 바꾼 ablation, metric만 추가한 재평가, 그림만 다시 그리는 수정이 각각 해당 모듈 하나만 건드리면, 실험 사이의 차이가 의도한 차이뿐이라고 자신할 수 있다.

:::note[분야 예시]
관측소 기온 자료로 강수를 예측하는 기후 연구를 생각한다. 데이터 처리는 결측 구간 보간과 단위 통일을, 모델은 시계열 신경망을, 평가는 지역별 RMSE와 극한값 재현율을, 시각화는 지도와 시계열 그림을 맡는다. 극한 강수 지표를 새로 추가하는 일은 평가 모듈만 바꾸면 되고, 이미 학습한 모델을 다시 학습할 필요가 없다.
:::

## 표준 디렉터리 구조

책임을 나누었으면 각 책임이 놓일 자리를 정한다. 많은 Python 연구 프로젝트가 쓰는 관례적인 구조는 다음과 같다. 공식 규격이 아니라 관례이므로 연구실마다 조금씩 다르지만, 뼈대는 대체로 같다.

```text
my-project/
├── README.md            # 목적 · 설치 · 실행 · 재현 방법
├── pyproject.toml       # 패키지 메타데이터와 의존성
├── configs/             # 실험 설정 파일 (6.3장)
├── src/
│   └── my_project/      # 설치 가능한 패키지: 재사용되는 모든 코드
│       ├── data/        # 읽기 · 전처리 · 분할 · Dataset
│       ├── models/      # 모델 정의
│       ├── training/    # 학습 loop, loss, optimizer 설정
│       ├── evaluation/  # metric, 평가 루틴
│       └── utils/       # 여러 책임이 공유하는 작은 도구
├── scripts/             # 진입점: train.py · eval.py · predict.py
├── tests/               # 자동 테스트 (6.6장)
├── notebooks/           # 탐색 · 분석
├── data/                # 데이터 위치 연결 (커밋하지 않음)
├── outputs/             # 실행 결과: outputs/<run-id>/ (커밋하지 않음)
└── docs/                # 설계 문서 · 결정 기록 (6.7장)
```

각 디렉터리의 역할은 다음과 같다.

- **`src/<project>/`** 는 재사용되는 코드가 사는 곳이다. 패키지로 설치하므로(`pip install -e .` 같은 editable install) 어느 디렉터리에서 실행하든 `import my_project`가 동작한다. `src/` 아래에 한 단계를 더 두는 src layout을 쓰면, 설치하지 않은 채 현재 디렉터리의 코드를 우연히 import하는 실수를 막을 수 있다. 이 차이는 5.2장에서 다뤘다.
- **`scripts/`** 는 사람이 실행하는 진입점이다. 인자와 설정을 읽고 `src`의 함수를 부르는 일만 한다.
- **`configs/`** 는 실행마다 바뀌는 값(데이터 경로, 모델 크기, 학습률, seed)을 코드 밖에 둔다. 6.3장의 주제다.
- **`tests/`** 는 shape·분할·metric 같은 불변조건을 자동으로 확인한다.
- **`data/`** 와 **`outputs/`** 는 저장소에 커밋하지 않는다. 아래 "데이터와 출력의 위치"에서 다룬다.

`src/<project>/` 안의 하위 패키지 이름은 앞 절의 책임과 거의 일대일로 대응한다. loss는 작으면 `training/` 안에, 커지면 `losses/`로 독립시킨다. 시각화는 그림을 만드는 함수가 여러 notebook과 스크립트에서 쓰이면 `plotting/` 같은 하위 패키지로 둔다.

:::caution[함정]
`utils/`는 "어디에 둘지 모르는 코드"의 쓰레기통이 되기 쉽다. 데이터 전처리 함수가 `utils/`에 있으면 데이터 처리 책임이 두 곳으로 흩어진다. `utils/`에는 특정 책임에 속하지 않는 작은 도구(경로 처리, 시간 측정, seed 설정 등)만 둔다.
:::

## 진입점

**진입점**(entry point)은 사람이 직접 실행하는 파일이다. 연구 프로젝트에는 보통 세 개가 있다.

| 진입점 | 입력 | 출력 |
|---|---|---|
| `scripts/train.py` | 설정 | checkpoint, 학습 로그, resolved config |
| `scripts/eval.py` | checkpoint, 평가 데이터 설정 | metric, 예측값 |
| `scripts/predict.py` | checkpoint, 새 입력 | 예측값 |

진입점은 **얇게** 쓴다. 인자를 읽고, 설정을 만들고, `src`의 함수를 부르고, 결과를 저장한다. 계산 로직이 진입점에 있으면 다른 진입점이나 테스트에서 재사용할 수 없다. 구조만 보이면 다음과 같다(`my_project`의 함수 이름은 예시다).

```python
# scripts/eval.py — 인자를 읽고 라이브러리 함수를 부르는 일만 한다
import argparse
from pathlib import Path

from my_project.data import load_split
from my_project.evaluation import evaluate, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="저장된 checkpoint를 평가한다.")
    parser.add_argument("--ckpt", type=Path, required=True, help="checkpoint 파일")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    model = load_checkpoint(args.ckpt)
    metrics = evaluate(model, load_split(args.split))
    print(metrics)


if __name__ == "__main__":
    main()
```

**평가 코드를 독립적으로 실행할 수 있어야 하는 이유**는 네 가지다.

- **재평가.** metric 계산에서 버그를 찾으면, 학습을 다시 하지 않고 저장된 checkpoint를 모두 다시 평가해야 한다.
- **비교.** 여러 checkpoint(다른 epoch, 다른 seed, 다른 baseline)를 **같은 평가 코드**로 비교해야 공정하다.
- **새 데이터.** 외부 test set이나 새 측정 조건에 같은 모델을 적용한다.
- **검증.** 학습 직후 기록한 숫자와 저장본으로 다시 계산한 숫자가 같은지 확인하면, 저장·불러오기 과정의 버그를 잡을 수 있다.

평가가 checkpoint와 데이터에만 의존한다는 것을 작은 예로 확인한다. `train`은 선형 회귀 계수를 dict로 반환하고, `evaluate`는 그 dict와 데이터만 받는다. checkpoint를 JSON 문자열로 저장했다가 다시 읽어 평가해도 같은 숫자가 나와야 한다.

```python {exec="browser" session="structure"}
import json
import numpy as np

def make_data(seed, n=200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 3))                               # (N, 3)
    y = x @ np.array([2.0, -1.0, 0.5]) + 0.1 * rng.normal(size=n)    # (N,)
    return x, y

def train(x, y):
    X = np.c_[x, np.ones(len(x))]                    # (N, 4) bias 열 추가
    w, *_ = np.linalg.lstsq(X, y, rcond=None)        # (4,)
    return {"w": w.tolist()}                         # checkpoint: 직렬화 가능한 dict

def evaluate(ckpt, x, y):
    w = np.array(ckpt["w"])                          # (4,)
    pred = np.c_[x, np.ones(len(x))] @ w             # (N,)
    return float(np.sqrt(np.mean((pred - y) ** 2)))

x_tr, y_tr = make_data(seed=0)
x_te, y_te = make_data(seed=1)
ckpt = train(x_tr, y_tr)
saved = json.dumps(ckpt)                             # 파일 대신 문자열로 저장
print(f"학습 직후 평가     RMSE = {evaluate(ckpt, x_te, y_te):.6f}")
print(f"저장본만으로 평가  RMSE = {evaluate(json.loads(saved), x_te, y_te):.6f}")
```

```output
학습 직후 평가     RMSE = 0.109791
저장본만으로 평가  RMSE = 0.109791
```

두 숫자가 같다. 실제 프로젝트에서는 저장 형식(정밀도, dtype), 모델의 평가 모드 전환, 데이터 전처리 통계(평균·표준편차)의 저장 여부 때문에 두 숫자가 어긋나는 일이 생긴다. 어긋난다면 checkpoint에 평가에 필요한 정보가 다 들어 있지 않다는 신호다.

::quiz{id="eval-independent"}

`scripts/` 아래 파일을 직접 실행하는 대신 `python -m my_project.train`처럼 패키지 모듈로 실행하거나, `pyproject.toml`에 명령 이름을 등록하는 방법도 있다. 진입점을 명령행 도구로 설계하는 방법은 6.4장에서 다룬다.

## notebook의 자리

**왜 필요한가.** notebook(Jupyter 등)은 탐색에 탁월하다. 데이터를 불러 바로 그려 보고, 셀 하나를 고쳐 다시 실행하고, 그림과 설명을 한 문서에 모은다. 문제는 notebook이 탐색 도구에서 **결과를 만드는 도구**로 바뀔 때 생긴다.

notebook의 약점은 셋이다.

- **숨은 상태.** 셀은 아무 순서로나 실행할 수 있고, 지운 셀이 만든 변수도 커널에 남는다. 화면에 보이는 코드와 실제 상태가 다를 수 있다.
- **재사용과 테스트의 어려움.** notebook의 함수는 다른 notebook이나 스크립트에서 import하기 어렵다. 그래서 복사하게 되고, 복사본은 서로 어긋난다.
- **리뷰의 어려움.** notebook 파일은 코드·출력·메타데이터가 섞인 JSON이라 diff를 읽기 어렵다.

숨은 상태가 어떻게 결과를 바꾸는지 본다. 아래 `normalize`는 전역 변수 `scale`을 몰래 읽는다. 같은 호출 `normalize(v)`가, 그 사이 다른 셀이 무엇을 실행했느냐에 따라 다른 값을 낸다.

```python {exec="browser" session="structure"}
scale = 1.0
def normalize(v):
    return v / scale              # 전역 변수 scale을 몰래 읽는다

v = np.array([2.0, 4.0])
print("셀 A를 실행한 직후:", normalize(v))
scale = v.max()                   # 아래쪽 셀에서 scale을 바꿨다
print("셀 B를 실행한 뒤  :", normalize(v))

def normalize_pure(v, scale):
    return v / scale              # 필요한 값은 모두 인자로 받는다

print("순수 함수         :", normalize_pure(v, 1.0), normalize_pure(v, v.max()))
```

```output
셀 A를 실행한 직후: [2. 4.]
셀 B를 실행한 뒤  : [0.5 1. ]
순수 함수         : [2. 4.] [0.5 1. ]
```

`normalize_pure`처럼 필요한 값을 모두 인자로 받는 **순수 함수**(pure function)는 호출만 보면 결과를 알 수 있다. `src/`로 옮기는 코드는 이런 형태여야 한다.

::quiz{id="notebook-hidden-state"}

**역할 분담.** notebook은 탐색과 분석, `src/`는 재사용과 결과 생산을 맡는다. 다음 중 하나라도 해당하면 notebook의 코드를 `src/`로 옮긴다.

1. 같은 코드를 두 번째 notebook에 복사하려 한다.
2. 그 코드가 논문·보고서에 들어갈 숫자나 그림을 만든다.
3. 그 코드가 틀리면 결론이 바뀌므로 테스트가 필요하다.

옮긴 뒤에는 notebook이 `src`의 함수를 import해서 쓴다. 패키지를 editable 모드로 설치해 두면 `src`의 코드를 고쳤을 때 다시 설치할 필요가 없다. 다만 이미 실행 중인 커널은 한 번 import한 모듈을 다시 읽지 않으므로, 커널을 재시작하거나 IPython의 `autoreload` 확장을 쓴다. 확장 사용법과 한계는 IPython 문서에서 확인한다.

notebook에 남는 분석도 최소한의 규칙을 지킨다. 결과를 저장하거나 공유하기 전에는 커널을 재시작하고 모든 셀을 위에서부터 다시 실행해, 화면의 결과가 화면의 코드에서 나왔음을 확인한다. 데이터 경로나 run ID 같은 입력은 notebook 맨 위 셀 한곳에 모은다.

## 데이터와 출력의 위치

**데이터.** 데이터는 Git 저장소에 넣지 않는다. 크기가 커서 저장소를 무겁게 만들고, 라이선스나 개인정보 문제로 공유할 수 없는 경우도 많다. 대신 `data/`에는 **데이터가 어디 있는지**를 연결한다.

- `data/README.md`에 출처, 내려받는 방법, 버전(받은 날짜, checksum)을 적는다.
- 공유 저장소에 있는 데이터는 `data/` 아래로 symbolic link를 건다.
- 원시 데이터(`data/raw/`)는 읽기만 하고, 전처리 결과는 `data/processed/`처럼 따로 둔다. 전처리 결과는 원시 데이터와 전처리 코드로 언제든 다시 만들 수 있어야 한다.

```bash
# 집필 시점 예시: 공유 스토리지의 데이터를 프로젝트에 연결한다(경로는 환경마다 다르다)
ln -s /shared/datasets/battery-cycles/v3 data/raw
ls -l data/                      # 링크가 가리키는 위치를 확인한다
```

코드는 데이터 위치를 직접 쓰지 않고 설정에서 받는다. `/home/kim/data/...` 같은 절대 경로가 코드에 들어가면 다른 사람·다른 서버에서 실행되지 않는다. 경로를 설정과 프로젝트 root로 다루는 방법은 6.3장에서 다룬다.

**출력.** 실행 결과는 `outputs/<run-id>/` 아래에 실행 하나당 디렉터리 하나로 모은다. 한 디렉터리에는 그 실행을 이해하고 다시 만드는 데 필요한 것이 모두 들어 있어야 한다.

```text
outputs/20261107-140309-9bb31a70/
├── config.json        # 실제로 적용된 설정 (6.3장)
├── train.log          # 학습 로그 (6.5장)
├── metrics.json       # 평가 결과
├── checkpoints/       # 모델 가중치
└── predictions/       # 예측값
```

`outputs/model.pt`처럼 고정된 이름에 덮어쓰면, 어느 실행의 결과인지 알 수 없고 이전 결과가 사라진다. run ID는 시작 시각과 설정의 hash를 합쳐 만들 수 있다. 같은 설정이면 키 순서가 달라도 같은 hash가 나오도록, 키를 정렬해 직렬화한 뒤 hash를 계산한다.

```python {exec="browser" session="structure"}
import hashlib
from pathlib import PurePosixPath

def make_run_id(config, started):
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canon.encode()).hexdigest()[:8]
    return f"{started}-{digest}"

cfg_a = {"model": {"width": 64}, "train": {"lr": 1e-3, "seed": 0}}
cfg_b = {"train": {"seed": 0, "lr": 1e-3}, "model": {"width": 64}}   # 키 순서만 다름
cfg_c = {"model": {"width": 64}, "train": {"lr": 1e-3, "seed": 1}}   # seed만 다름
for name, cfg in [("a", cfg_a), ("b", cfg_b), ("c", cfg_c)]:
    print(name, make_run_id(cfg, "20261107-140309"))

run_dir = PurePosixPath("outputs") / make_run_id(cfg_a, "20261107-140309")
for f in ["config.json", "train.log", "metrics.json", "checkpoints/last.npz"]:
    print(run_dir / f)
```

```output
a 20261107-140309-9bb31a70
b 20261107-140309-9bb31a70
c 20261107-140309-1c7df5ed
outputs/20261107-140309-9bb31a70/config.json
outputs/20261107-140309-9bb31a70/train.log
outputs/20261107-140309-9bb31a70/metrics.json
outputs/20261107-140309-9bb31a70/checkpoints/last.npz
```

키 순서만 다른 `a`와 `b`는 같은 hash, seed만 다른 `c`는 다른 hash를 받는다. hash가 같은 디렉터리끼리는 "같은 설정을 다른 시각에 다시 실행한 것"으로 묶을 수 있다. 시각을 앞에 두면 디렉터리 이름 정렬이 곧 시간순이 된다.

::quiz{id="run-id-config-hash"}

`data/`(링크와 README만 남기고)와 `outputs/`는 `.gitignore`에 넣는다. 논문에 쓸 최종 결과처럼 오래 보존할 출력은 Git이 아니라 별도 저장소(연구실 스토리지, 데이터 저장소)에 올리고, 그 위치를 문서에 남긴다. 결과 관리 도구와 운영은 22.1장에서 다룬다.

:::note[분야 예시]
계산화학에서 분자마다 DFT 계산을 돌리면 입력 구조, 계산 설정, 출력 파일, 추출한 에너지가 쌓인다. 계산 하나를 `outputs/<run-id>/`에 대응시키고, 계산 설정(범함수, 기저 함수, 수렴 기준)을 그 디렉터리의 설정 파일로 남기면, 나중에 "이 에너지는 어떤 설정으로 계산했는가"를 디렉터리만 보고 답할 수 있다. 원시 출력이 크면 공유 스토리지에 두고 `data/`에서 연결한다.
:::

## 추상화의 균형

코드를 나누다 보면 반대 방향의 두 실패를 만난다.

- **중복.** `train.py`를 복사해 `train_v2.py`, `train_lstm.py`, `train_final.py`를 만든다. 데이터 로딩의 버그를 한 파일에서만 고치면, 나머지 실험은 버그가 있는 전처리로 계속 돌아간다. 결과 비교가 공정하지 않게 된다.
- **과도한 추상화.** 모델이 하나뿐인데 registry·factory·plugin 구조를 먼저 만든다. 새 실험 하나를 추가하려면 다섯 파일을 고쳐야 하고, 코드를 처음 보는 사람은 실제로 무엇이 실행되는지 추적하지 못한다.

연구 코드는 아직 무엇이 바뀔지 모르는 상태에서 쓰인다. 그래서 공통화의 시점이 중요하다. 실험 코드가 공통 모듈로 **승격**되는 기준은 다음과 같다.

| 신호 | 판단 |
|---|---|
| 같은 코드가 세 번째로 필요해졌다 | 같은 부분과 달라지는 부분을 볼 수 있으므로 승격한다 |
| 두 사례만 있다 | 아직 기다린다. 무엇이 공통인지 두 점으로는 알기 어렵다 |
| 코드가 결과의 공정성에 영향을 준다(분할, metric) | 사례 수와 상관없이 하나로 만든다. 실험마다 다르면 비교가 무의미하다 |
| 인터페이스가 실험마다 계속 바뀐다 | 아직 실험 스크립트에 둔다 |
| 테스트를 붙일 수 있을 만큼 동작이 정해졌다 | 승격하고 테스트를 함께 옮긴다 |

흔한 진화 경로는 notebook의 셀 → `scripts/`의 실험별 스크립트 → `src/`의 공통 함수다. 각 단계로 옮길 때 이름과 인자를 다듬고, `src/`로 들어갈 때 테스트를 붙인다.

::quiz{id="premature-abstraction"}

:::note[분야 예시]
단백질 서열을 다루는 생명과학 연구실에서, 서열 정제(비표준 아미노산 처리, 길이 필터, 중복 제거) 코드가 세 프로젝트에 복사되어 있었다고 하자. 세 복사본의 길이 기준이 조금씩 다르면, 세 프로젝트의 결과는 서로 다른 데이터에서 나온 셈이 된다. 이 코드는 연구실 공통 패키지로 승격할 신호다. 반면 프로젝트마다 다른 구조 예측 모델은 각 프로젝트에 남긴다.
:::

## 실습

연구실 공통 템플릿으로 프로젝트를 시작하고, 기존 notebook 코드를 구조에 맞게 옮긴다. 템플릿을 만드는 스크립트는 `labs/research-code-engineering/project-structure/scaffold.py`다. 지정한 디렉터리에 표준 구조와 최소 파일(진입점, 설정, smoke test)을 만들고, 이미 있는 파일은 덮어쓰지 않는다. 집필 시점의 실행 결과는 다음과 같다.

```bash
uv run --project labs python labs/research-code-engineering/project-structure/scaffold.py soh-project --name soh_model
```

```output
soh-project/  (새 파일 16개)
├── configs/
│   └── default.toml
├── data/
│   └── README.md
├── docs/
│   └── README.md
├── notebooks/
│   └── README.md
├── outputs/
├── scripts/
│   ├── eval.py
│   └── train.py
├── src/
│   └── soh_model/
│       ├── data/
│       │   └── __init__.py
│       ├── evaluation/
│       │   └── __init__.py
│       ├── models/
│       │   └── __init__.py
│       ├── training/
│       │   └── __init__.py
│       ├── utils/
│       │   └── __init__.py
│       └── __init__.py
├── tests/
│   └── test_smoke.py
├── .gitignore
├── README.md
└── pyproject.toml
```

만든 프로젝트 안에서 `uv run --with pytest pytest -q`를 실행하면 패키지가 설치되고 smoke test(`import soh_model`) 하나가 통과한다(집필 시점 예시. 패키지 관리 도구의 명령은 3부에서 확인한다). 연구실에 이미 템플릿 저장소가 있다면 그것을 쓴다. 템플릿을 복제하는 방법(저장소 템플릿 기능, 템플릿 생성 도구 등)은 연구실 문서와 도구 문서에서 확인한다.

이어서 다음 순서로 기존 코드를 옮긴다.

1. 기존 notebook의 함수를 책임별로 분류하고(데이터·모델·loss·학습·평가·시각화), 해당하는 `src/soh_model/` 하위 패키지로 옮긴다. 전역 변수를 읽는 함수는 인자를 받도록 고친다.
2. `scripts/train.py`가 checkpoint와 학습 직후 metric을 `outputs/<run-id>/`에 저장하게 한다.
3. `scripts/eval.py`가 checkpoint 경로만 받아 같은 metric을 다시 계산하게 한다.
4. notebook은 `src`의 함수를 import해 그림만 그리도록 줄인다. 커널을 재시작하고 모든 셀을 다시 실행해 같은 그림이 나오는지 확인한다.

run ID를 만드는 함수를 연습문제로 구현한다. 결과에 영향을 주지 않는 설정(출력 위치, 로그 수준)은 hash에서 빼야 한다.

::exercise{id="run-id"}

## 실패 사례

- **모든 것이 하나의 거대한 `train.py`에 있는 것.** 데이터 처리, 모델, 학습, 평가, 그림이 한 파일에 있으면 그림 하나를 고치려 해도 학습부터 다시 돌린다. 실험마다 이 파일을 복사하기 시작하면 복사본 사이의 차이를 아무도 추적하지 못한다.
- **평가 코드가 학습 코드 안에서만 실행되는 것.** metric 계산의 버그를 고쳐도 이미 학습한 모델을 다시 평가할 방법이 없다. 결국 학습을 다시 돌리거나, 버그가 있는 숫자를 그대로 보고한다.
- **결과를 만드는 계산이 notebook에만 있는 것.** 셀 실행 순서에 따라 숫자가 달라질 수 있고, 같은 숫자를 다시 만들려면 그 notebook을 "올바른 순서로" 실행하는 방법을 기억해야 한다.
- **`outputs/`에 고정된 이름으로 덮어쓰는 것.** 어느 실행이 지금의 `model.pt`를 만들었는지 알 수 없다.

## 통과 기준

- 학습과 독립적으로 평가 스크립트만 실행해, 학습 직후 기록한 metric과 같은 값을 재현한다.
- 템플릿으로 만든 프로젝트를 설치하고 smoke test를 통과시킨다. 기존 notebook의 재사용 코드가 `src/`의 책임별 하위 패키지로 옮겨져 있다.
- 연습문제 `run-id`의 테스트를 통과한다.
