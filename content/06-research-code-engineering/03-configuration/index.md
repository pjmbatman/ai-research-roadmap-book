---
id: research-code-engineering/configuration
title: "설정 관리"
title_en: "Configuration Management"
tier: core
status: draft
prerequisites: [research-code-engineering/project-structure]
---

# 설정 관리

연구는 같은 코드를 다른 값으로 여러 번 실행하는 일이다. 학습률을 바꾸고, seed를 바꾸고, 데이터셋과 모델 크기를 바꾼다. 이 값들이 코드 곳곳에 숫자로 박혀 있으면, 실험할 때마다 코드를 고쳐야 하고, 어떤 결과가 어떤 값에서 나왔는지는 기억에 의존하게 된다. 석 달 뒤 논문 표의 숫자 하나를 다시 만들어야 할 때 그 기억은 남아 있지 않다.

이 장은 코드와 설정을 분리하고, 기본값·override·검증을 설계하고, 실제로 적용된 설정을 결과 옆에 저장하는 방법을 다룬다. 목표는 어떤 결과 파일을 집어 들어도 그 결과를 만든 정확한 설정을 복원할 수 있는 상태다.

## 학습 목표

- 코드와 설정을 분리한다
- 설정의 기본값·override·검증을 설계한다
- 실제 적용된 설정을 저장한다

## 왜 설정을 분리하는가

**왜 필요한가.** 학습률을 바꾸려고 `train.py`의 `lr = 1e-3`을 `3e-4`로 고쳐 실행했다고 하자. 실행이 끝나면 다시 되돌리거나 그대로 둔다. 어느 쪽이든, 그 실행이 어떤 학습률을 썼는지는 코드 어디에도 남지 않는다. 여러 실험을 동시에 돌리려면 코드 사본이 여러 개 필요해진다. 코드 수정과 실험 조건 변경이 한데 섞여 Git 이력도 읽을 수 없게 된다.

**정의.** **설정**(configuration)은 실행마다 달라질 수 있는 값이고, **코드**는 실행마다 같은 로직이다. 경계를 정하는 질문은 "이 값을 바꾸려고 코드를 고친 적이 있는가, 또는 고칠 것 같은가"다. 연구 프로젝트에서 설정으로 빼는 값은 대개 네 묶음이다.

| 묶음 | 예 |
|---|---|
| data | 데이터셋 이름, 경로, 분할 방식과 비율, 전처리 옵션 |
| model | 모델 종류, 폭·깊이, dropout |
| train | 학습률, epoch 수, batch 크기, optimizer, seed |
| eval | metric 목록, 평가할 checkpoint, 평가 데이터 |

반대로 코드에 두는 값도 있다. 물리 상수(Faraday 상수 등), 알고리즘의 정의 자체, 실험 조건이 아닌 구현 세부(예: 수치 안정성을 위한 작은 $\epsilon$이 결과에 영향을 주지 않는 경우)다. 모든 숫자를 설정으로 빼면 설정 파일이 코드만큼 복잡해진다. API 키나 비밀번호 같은 비밀 값은 설정 파일에도 넣지 않는다. 설정 파일은 결과와 함께 저장·공유되기 때문이다. 비밀 값은 환경 변수나 비밀 관리 도구로 전달한다.

**연구에서의 의미.** 설정이 분리되면 "실험 하나 = 설정 하나"가 된다. 코드 한 벌로 여러 설정을 동시에 실행할 수 있고, 두 실험의 차이는 두 설정 파일의 diff로 정확히 드러난다. ablation, hyperparameter 탐색, seed 반복이 모두 설정 목록을 만드는 일로 바뀐다.

:::note[분야 예시]
기후·기상 수치 모델은 오래전부터 격자 해상도, 적분 시간 간격, 물리 과정 선택 같은 실행 조건을 namelist라는 별도 텍스트 파일로 분리해 왔다. 같은 실행 파일로 수많은 조건의 시뮬레이션을 돌리고, 그 파일을 결과와 함께 보관해 어떤 조건의 결과인지 기록한다. 머신러닝 연구의 설정 파일도 같은 역할을 한다.
:::

## 설정 형식

설정 파일 형식으로는 YAML, TOML, JSON이 널리 쓰인다.

| 형식 | 주석 | 중첩 | Python에서 읽기 | 특징 |
|---|---|---|---|---|
| JSON | 없음 | 객체 | 표준 라이브러리 `json` | 엄격하고 단순하다. 사람이 쓰기에는 불편하다 |
| TOML | `#` | `[section]` 표 | 표준 라이브러리 `tomllib`(Python 3.11부터, 읽기 전용) | 타입이 명확하다. 깊은 중첩은 장황해진다 |
| YAML | `#` | 들여쓰기 | 서드파티 라이브러리 필요 | 간결하다. 값의 타입을 추측하는 규칙에 함정이 있다 |

어떤 형식이든 최상위 구조는 앞 절의 네 묶음을 따른다. TOML로 쓴 설정을 읽어 각 값의 타입을 확인한다.

```python {exec="browser" session="config"}
import json
import tomllib

text = '''
[data]
name = "cycles-v3"
root = "data/processed"

[model]
width = 64
depth = 3

[train]
lr = 1e-3
epochs = 50
seed = 0

[eval]
metrics = ["rmse", "mae"]
'''
cfg = tomllib.loads(text)
for section, values in cfg.items():
    print(f"{section:5s}", {k: type(v).__name__ for k, v in values.items()})
print(json.dumps(cfg["train"]))
```

```output
data  {'name': 'str', 'root': 'str'}
model {'width': 'int', 'depth': 'int'}
train {'lr': 'float', 'epochs': 'int', 'seed': 'int'}
eval  {'metrics': 'list'}
{"lr": 0.001, "epochs": 50, "seed": 0}
```

TOML은 `1e-3`을 실수로, `50`을 정수로, 따옴표 안의 값을 문자열로 읽는다. 타입이 표기에서 바로 정해진다.

YAML은 따옴표 없는 값의 타입을 문자열 모양으로 추측한다. 그런데 YAML 1.1 규칙을 따르는 로더에서는 이 추측이 연구자의 기대와 다를 때가 있다. 다음은 PyYAML(집필 시점 6.0.3)로 확인한 결과다.

```python
import yaml
yaml.safe_load("lr: 1e-3\nlr2: 1.0e-3\nflag: yes\nversion: 3.10\nnorway: NO")
```

```output
{'lr': '1e-3', 'lr2': 0.001, 'flag': True, 'version': 3.1, 'norway': False}
```

`1e-3`은 소수점이 없어 문자열이 되고, `yes`와 `NO`는 불리언이 되고, 버전 번호 `3.10`은 실수 3.1이 된다. 쓰는 라이브러리와 버전에 따라 규칙이 다를 수 있으므로, YAML을 쓴다면 이런 값을 직접 로드해 타입을 확인한다. 어떤 형식을 쓰든 결론은 같다. **읽은 값의 타입을 믿지 말고 검증한다.** 다음 절의 주제다.

## 설정 객체

**왜 필요한가.** 읽은 설정을 dict 그대로 코드에 넘기면 세 가지 문제가 생긴다. 키 이름의 오타는 그 키를 실제로 읽는 순간에야 `KeyError`로 드러난다. 학습이 몇 시간 돈 뒤일 수도 있다. `cfg.get("dropout", 0.1)`처럼 코드 곳곳에 기본값을 두면, 설정 파일에 없는 키가 어떤 값으로 실행됐는지 설정만 봐서는 알 수 없다. 그리고 타입과 범위가 틀린 값이 그대로 계산에 들어간다.

**정의.** **설정 객체**는 설정의 키·타입·기본값·제약을 한곳에 선언한 객체다. 설정 파일을 읽은 직후 dict를 설정 객체로 바꾸고, 이후의 코드는 설정 객체만 받는다. 이 변환 단계에서 다음을 검사한다.

- **알 수 없는 키**: 선언에 없는 키(대개 오타)는 거부한다.
- **누락된 키**: 기본값이 없는 필수 키가 빠졌으면 거부한다.
- **타입과 범위**: 학습률은 0과 1 사이의 실수, epoch 수는 양의 정수.
- **충돌**: 키 사이의 조건. 예를 들어 분할 비율의 합은 1이어야 하고, 평가 모드에서는 checkpoint 경로가 필요하다.

Python 표준 라이브러리의 `dataclass`로 설정 객체를 만들 수 있다. `frozen=True`로 만들면 실행 중에 설정을 바꿀 수 없어, 저장한 설정과 실제로 쓴 설정이 어긋나는 일을 막는다. 범위 검사는 `__post_init__`에 쓴다.

```python {exec="browser" session="config"}
from dataclasses import dataclass, fields

@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 50
    seed: int = 0

    def __post_init__(self):
        if not 0 < self.lr < 1:
            raise ValueError(f"train.lr는 (0, 1) 범위여야 한다: {self.lr}")

def from_dict(cls, d, section):
    unknown = set(d) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"{section}에 알 수 없는 키 {sorted(unknown)}")
    return cls(**d)

for raw in [{"lr": 3e-4}, {"learning_rate": 3e-4}, {"lr": 3.0}, {"seed": "0"}]:
    try:
        print("통과:", from_dict(TrainConfig, raw, "train"))
    except ValueError as e:
        print("거부:", e)
```

```output
통과: TrainConfig(lr=0.0003, epochs=50, seed=0)
거부: train에 알 수 없는 키 ['learning_rate']
거부: train.lr는 (0, 1) 범위여야 한다: 3.0
통과: TrainConfig(lr=0.001, epochs=50, seed='0')
```

오타(`learning_rate`)와 범위 밖의 값(`3.0`)은 실행 전에 거부된다. 그런데 마지막 줄을 보면 문자열 `'0'`이 정수 필드 `seed`를 통과했다. dataclass의 type hint는 **실행 중에 검사되지 않는다**. 타입까지 검증하려면 `__post_init__`에서 `isinstance`로 직접 검사하거나(실습 코드의 `build` 함수가 이렇게 한다), 실행 중 타입 검증과 변환을 해 주는 라이브러리를 쓴다. pydantic이 대표적이다(집필 시점 예시. 검증 규칙과 변환 동작은 라이브러리 문서에서 확인한다).

::quiz{id="dataclass-no-type-check"}

설정 객체의 선언은 그 자체로 문서가 된다. 어떤 키가 있고, 각각의 타입과 기본값이 무엇인지 한 파일에서 읽을 수 있다. 기본값을 코드 곳곳의 `.get(..., 기본값)`이 아니라 이 선언 한곳에만 두는 것이 핵심이다.

## 조합과 override

실험 설정은 대부분 기본 설정에서 몇 값만 바꾼 것이다. 그래서 설정을 **층**(layer)으로 쌓는다.

1. **기본 설정**(`configs/default.toml`): 모든 키와 기본값.
2. **실험 파일**(`configs/<실험>.toml`): 그 실험에서 바꾸는 값만.
3. **명령행 override**(`train.lr=3e-4`): 이번 실행에서만 바꾸는 값.

뒤의 층이 앞의 층을 덮어쓴다. 우선순위는 기본값 < 실험 파일 < 명령행이다. 이때 중첩 dict는 **깊게** 병합해야 한다. Python의 dict 펼치기 `{**a, **b}`는 최상위 키만 병합한다.

```python {exec="browser" session="config"}
import copy

defaults = {"model": {"width": 64, "depth": 3}, "train": {"lr": 1e-3, "seed": 0}}
exp = {"model": {"width": 128}}                  # 실험 파일: 바꿀 값만 쓴다
print("얕은 병합:", {**defaults, **exp})

def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out

def parse_override(item):                        # "train.lr=3e-4" → {"train": {"lr": 0.0003}}
    dotted, raw = item.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]  # TOML 문법으로 타입을 해석한다
    except tomllib.TOMLDecodeError:
        value = raw                              # 따옴표 없는 문자열
    for key in reversed(dotted.split(".")):
        value = {key: value}
    return value

merged = deep_merge(defaults, exp)
for item in ["train.lr=3e-4", "train.seed=1"]:
    merged = deep_merge(merged, parse_override(item))
print("깊은 병합:", merged)
```

```output
얕은 병합: {'model': {'width': 128}, 'train': {'lr': 0.001, 'seed': 0}}
깊은 병합: {'model': {'width': 128, 'depth': 3}, 'train': {'lr': 0.0003, 'seed': 1}}
```

얕은 병합은 `model.depth`를 잃었다. 이후 코드가 `.get("depth", 2)`처럼 기본값을 두고 읽는다면, 오류 없이 의도와 다른 깊이로 실행된다. 깊은 병합은 실험 파일이 바꾼 `width`만 바꾸고 나머지를 유지한다. 명령행 override는 점 경로(dotted key)로 위치를 가리키고, 값은 TOML 문법으로 해석해 `3e-4`는 실수, `1`은 정수가 되게 했다.

::quiz{id="shallow-merge"}

위의 `deep_merge`에는 아직 구멍이 있다. 기본 설정에 없는 키(`train.learning_rate`)를 덮어쓰면 새 키로 조용히 추가된다. 코드는 계속 `train.lr`을 읽으므로 그 override는 아무 효과가 없다. 설정 객체로 바꿀 때 알 수 없는 키를 거부하거나, 병합 단계에서부터 거부해야 한다. 실습의 연습문제가 후자를 구현한다.

**OmegaConf와 Hydra의 개념.** 이런 층 쌓기를 일반화한 도구가 있다(집필 시점 예시. 기능과 문법은 각 도구의 문서에서 확인한다).

- **OmegaConf**는 계층형 설정 객체 라이브러리다. 설정의 병합, 점 경로 형식의 명령행 override, 설정 안에서 다른 값을 참조하는 보간(interpolation), dataclass로 구조와 타입을 선언하는 기능을 제공한다.
- **Hydra**는 OmegaConf 위에서 동작하는 실험 설정 프레임워크다. 설정을 묶음(config group, 예: 모델 종류별 설정 파일)으로 나누어 조합하고, 명령행에서 묶음과 값을 바꾸고, 여러 조합을 한 번에 실행(sweep)하고, 실행마다 출력 디렉터리를 만들어 조합된 설정을 저장하는 일을 자동화한다.

도구를 쓰든 직접 구현하든 개념은 이 절과 같다. 층, 우선순위, 깊은 병합, 점 경로 override, 그리고 검증이다. 도구를 도입하기 전에 이 개념을 직접 구현해 보면 도구가 무엇을 대신하는지, 어디서 놀랄 수 있는지를 알 수 있다.

:::note[분야 예시]
DFT 계산으로 재료의 형성 에너지를 구하는 계산화학 연구에서, 범함수·기저 함수·에너지 수렴 기준·k-point 격자는 모든 계산에 공통인 기본 설정이고, 원소 조성과 구조 파일만 계산마다 다르다. 이를 기본 설정 + 계산별 override로 나누면, 수렴 기준을 바꾼 재계산 묶음과 원래 묶음의 차이가 설정 diff 한 줄로 드러난다.
:::

## 기록

**왜 필요한가.** 설정 파일을 분리해도, 실행이 실제로 어떤 값을 썼는지는 별도로 남겨야 한다. 실제 값은 기본 설정, 실험 파일, 명령행 override, 코드 속 기본값이 합쳐진 결과이고, 그중 어느 것도 혼자서는 그 실행을 설명하지 못한다.

**정의.** **resolved config**는 모든 층을 병합하고 검증을 마친 최종 설정이다. 실행 하나는 다음을 함께 기록한다.

- **resolved config**: 실제로 쓴 모든 값.
- **run ID**: 실행을 가리키는 이름. 6.1장처럼 시작 시각과 설정 hash로 만들 수 있다. 결과에 영향을 주지 않는 값(출력 위치 등)은 hash에서 뺀다.
- **코드 revision**: 커밋 해시(`git rev-parse HEAD`)와, 커밋되지 않은 변경이 있었는지(`git status --porcelain`). 변경이 있었다면 커밋 해시만으로는 그 코드를 복원할 수 없다.
- **데이터 버전**과 **실행 환경**: 데이터 checksum이나 버전 태그, 패키지 버전. 환경 기록은 3부, 재현성 전반은 22.2장에서 다룬다.

**언제 저장하는가.** 모든 병합과 검증이 끝난 직후, 학습을 시작하기 **전에** 저장한다. 실패하거나 중단된 실행의 설정이야말로 원인을 찾는 데 필요하다. 그리고 저장한 그 객체를 실행에 넘긴다. 설정을 저장한 뒤 코드가 설정을 고친다면 저장본은 거짓이 된다. `frozen=True`가 이를 막는다.

저장 시점을 잘못 고르면 어떤 일이 생기는지 본다. `wrong`은 명령행 override를 적용하기 전의 설정을 저장했고, `right`는 실제로 실행에 쓴 설정을 저장했다. 저장된 설정으로 다시 실행해 같은 결과가 나오는지 확인한다.

```python {exec="browser" session="config"}
def run(cfg):                                    # 학습 대신: 설정에 따라 정해지는 결과
    return {"rmse": round(0.1 + 0.01 * cfg["train"]["seed"], 4)}

base = deep_merge(defaults, exp)
final = deep_merge(base, parse_override("train.seed=2"))
wrong = {"config": base, "metrics": run(final)}      # override 전에 저장했다
right = {"config": final, "metrics": run(final)}     # 실제로 쓴 설정을 저장했다

for name, record in [("wrong", wrong), ("right", right)]:
    restored = json.loads(json.dumps(record))        # 파일로 썼다가 다시 읽기
    same = run(restored["config"]) == restored["metrics"]
    print(f"{name}: 저장된 설정으로 다시 실행해 같은 결과인가? {same}")
```

```output
wrong: 저장된 설정으로 다시 실행해 같은 결과인가? False
right: 저장된 설정으로 다시 실행해 같은 결과인가? True
```

`wrong`의 저장본은 seed 0을 가리키지만 결과는 seed 2에서 나왔다. 이런 기록은 기록이 없는 것보다 나쁘다. 틀린 설정을 믿고 재현을 시도하게 만들기 때문이다.

::quiz{id="resolved-config-timing"}

**결과 파일에서 설정으로.** 결과 파일(metric, 예측값, 그림)은 resolved config와 같은 run 디렉터리에 둔다. 결과가 run 디렉터리 밖으로 나가는 경우(논문용 표로 모은 CSV, 발표 자료의 그림)에는 run ID를 열이나 메타데이터로 함께 옮긴다. 그러면 어떤 숫자에서든 run ID → run 디렉터리 → resolved config와 코드 revision으로 되돌아갈 수 있다. JSON으로 저장할 때는 tuple이 list로 바뀌고 dict의 정수 키가 문자열로 바뀌므로, 복원 경로(실습의 `--replay`처럼)를 한 번 실제로 실행해 확인해 둔다.

## 경로 처리

**왜 필요한가.** 같은 코드가 프로젝트 root에서는 동작하고 notebook이나 다른 디렉터리에서는 `FileNotFoundError`를 내는 일은 흔하다. 원인은 대개 상대 경로다. `open("data/train.csv")`의 상대 경로는 코드가 있는 파일이 아니라 **현재 작업 디렉터리**(current working directory)를 기준으로 해석된다. notebook 커널, 작업 스케줄러, IDE의 실행 버튼은 각자 다른 작업 디렉터리에서 코드를 실행한다.

**경로의 기준.** 상대 경로를 해석할 기준은 세 가지가 있다.

| 기준 | 얻는 방법 | 성질 |
|---|---|---|
| 작업 디렉터리 | `Path.cwd()` | 실행하는 방법에 따라 달라진다 |
| 코드 파일의 위치 | `Path(__file__).resolve().parent` | 파일을 옮기면 달라진다 |
| 프로젝트 root | 표식 파일을 찾아 위로 올라감 | 실행 위치와 무관하다 |

연구 프로젝트에서는 **프로젝트 root**를 기준으로 삼는 것이 안정적이다. root는 `pyproject.toml`이나 `configs/` 같은 표식이 있는 디렉터리를 위로 올라가며 찾는다. 실습 코드의 함수는 다음과 같다.

```python
def find_project_root(start: Path, marker: str = "configs/default.toml") -> Path:
    """start에서 위로 올라가며 marker 파일이 있는 디렉터리를 찾는다. 작업 디렉터리와 무관하다."""
    for d in [start, *start.parents]:
        if (d / marker).is_file():
            return d
    raise FileNotFoundError(f"{start}의 상위 어디에도 {marker}가 없습니다")
```

**규칙.** 설정 파일에는 root 기준 상대 경로(`data/processed`)를 쓰고, 코드는 설정을 읽은 직후 `root / path`로 절대 경로를 만든다. `pathlib`에서 `root / path`는 `path`가 절대 경로이면 `path`를 그대로 돌려주므로, 클러스터의 scratch 디렉터리처럼 저장소 밖의 위치도 명령행 override(`paths.output_root=/scratch/me/outputs`)로 지정할 수 있다. 사람마다 다른 데이터 위치는 설정 파일을 고치는 대신 환경 변수나 개인용 override 파일(커밋하지 않음)로 넘긴다. 어느 경우든 resolved config에는 최종 절대 경로가 남는다.

## 실습

하나의 설정 파일과 명령행 override로 데이터셋·모델·seed·학습률을 바꾸어 실행하고, 각 실행의 resolved config를 저장한다. 코드는 `labs/research-code-engineering/configuration/`에 있다.

- `configs/default.toml`: 모든 키와 기본값(데이터 `sine`, 다항식 모델, 학습률 0.1, seed 0).
- `configs/linear-baseline.toml`: 모델만 선형으로 바꾸는 실험 파일.
- `run_experiment.py`: 기본 설정 → 실험 파일(`--exp`) → 명령행(`--set key=value`, 여러 번 가능) 순서로 병합하고, dataclass 설정 객체로 바꾸며 알 수 없는 키·빠진 키·타입·범위를 검증한다. 그다음 `outputs/<run-id>/config.json`에 resolved config와 코드 revision을 먼저 저장하고 실행해 `metrics.json`을 남긴다. `--replay <run 디렉터리>`는 저장된 설정만으로 다시 실행해 metric을 비교한다.

집필 시점의 실행 결과는 다음과 같다(run ID 앞부분의 시각은 실행할 때마다 다르다).

```bash
cd labs/research-code-engineering/configuration
uv run --project ../.. python run_experiment.py
uv run --project ../.. python run_experiment.py --exp configs/linear-baseline.toml --set train.seed=1
uv run --project ../.. python run_experiment.py --set model.degree=9 --set train.lr=0.3
```

```output
run 20261107-093706-01302a1a  model=poly seed=0 lr=0.1  {'test_rmse': 0.143698}
저장: outputs/20261107-093706-01302a1a/config.json, metrics.json
run 20261107-093706-908bfe21  model=linear seed=1 lr=0.1  {'test_rmse': 0.466013}
저장: outputs/20261107-093706-908bfe21/config.json, metrics.json
run 20261107-093706-49b992ab  model=poly seed=0 lr=0.3  {'test_rmse': 0.109019}
저장: outputs/20261107-093706-49b992ab/config.json, metrics.json
```

두 번째 실행의 `config.json`에는 어떤 층에서 무엇을 바꿨는지(`sources`), 코드 revision(`code`), 최종 설정(`config`)이 함께 남는다. `dirty: true`는 커밋되지 않은 변경이 있는 상태에서 실행했다는 뜻이다. 결과로 보고할 실행이라면 이 값이 `false`여야 한다.

```output
{
  "run_id": "20261107-093706-908bfe21",
  "sources": {"exp": "configs/linear-baseline.toml", "set": ["train.seed=1"]},
  "code": {"commit": "4234bd4858daebbf8f6a3e4046c134730624179c", "dirty": true},
  "config": {
    "data": {"name": "sine", "n_train": 200, "noise": 0.1},
    "model": {"name": "linear", "degree": 5},
    "train": {"lr": 0.1, "epochs": 2000, "seed": 1},
    "paths": {"output_root": "outputs"}
  }
}
```

(실제 파일은 한 줄에 키 하나씩 들여쓰기되어 있다. 여기서는 짧게 묶어 옮겼다.) 저장된 설정만으로 다시 실행하면 같은 metric이 나온다. 다른 작업 디렉터리에서 실행해도 결과 위치가 같은지 확인하기 위해, 한 단계 위 디렉터리에서 실행했다.

```bash
cd labs/research-code-engineering
uv run --project .. python configuration/run_experiment.py --replay outputs/20261107-093706-908bfe21
```

```output
저장된 metric : {'test_rmse': 0.466013}
다시 실행     : {'test_rmse': 0.466013}
일치          : True
```

설정 오타와 타입 오류는 실행 전에 거부된다. 다음 두 명령의 출력에서 traceback의 마지막 줄만 옮겼다.

```bash
uv run --project .. python configuration/run_experiment.py --set train.learning_rate=0.3
uv run --project .. python configuration/run_experiment.py --set train.epochs=1e3
```

```output
ValueError: train: 알 수 없는 키 ['learning_rate']
TypeError: train.epochs: int여야 합니다. 받은 값 1000.0
```

이어서 직접 해 볼 것:

1. `--set data.name=linear`, `--set model.degree=3` 등으로 데이터셋·모델·seed·학습률을 바꾼 실행을 네 개 이상 만들고, 각 run 디렉터리의 `config.json` 두 개를 diff해 차이가 의도한 값뿐인지 확인한다.
2. `configs/default.toml`의 학습률을 바꾼 뒤 이전 run을 `--replay`한다. 저장된 resolved config를 쓰므로 결과가 그대로여야 한다.
3. 연습문제로 기본 설정에 없는 키를 거부하는 깊은 병합 함수를 구현한다.

::exercise{id="strict-deep-merge"}

## 실패 사례

- **코드 곳곳에 하드코딩된 숫자와 경로.** `lr = 1e-3`이 학습 스크립트에, `/home/kim/data/cycles.csv`가 데이터 모듈에, 0.8이라는 분할 비율이 다른 함수에 흩어져 있다. 실험 조건을 바꾸려면 코드를 고쳐야 하고, 그 실행의 조건은 어디에도 남지 않는다. 절대 경로는 다른 사람·다른 서버에서 바로 실패한다.
- **저장된 설정과 실제 실행 설정이 다른 것.** 기본 설정 파일만 복사해 두고 명령행 override를 빠뜨리거나, 설정을 저장한 뒤 코드가 설정을 고치거나(예: GPU 메모리에 맞춰 batch 크기를 줄임), 코드 속 `.get(키, 기본값)`이 설정 파일에 없는 값을 채운다. 저장본을 믿고 재현하면 다른 결과가 나온다.
- **오타가 조용히 무시되는 것.** `train.learning_rate=3e-4`를 넘겼는데 코드는 `train.lr`을 읽는다. 실험자는 학습률을 바꿨다고 믿고 결과를 해석한다.
- **얕은 병합.** 실험 파일이 섹션 하나의 키 하나만 바꿨는데 그 섹션의 다른 키가 모두 사라진다.

## 통과 기준

- 임의의 결과 파일에서 run ID를 따라가 그 결과를 만든 정확한 설정(resolved config)과 코드 revision을 찾고, 그 설정으로 다시 실행해 같은 결과를 얻는다.
- 설정 오타·누락·타입 오류가 학습 시작 전에 원인을 가리키는 메시지와 함께 실패한다.
- 연습문제 `strict-deep-merge`의 테스트를 통과한다.
