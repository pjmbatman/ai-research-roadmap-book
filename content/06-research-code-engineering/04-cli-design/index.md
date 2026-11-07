---
id: research-code-engineering/cli-design
title: "CLI 설계"
title_en: "Designing Command-Line Interfaces"
tier: core
status: draft
prerequisites: [research-code-engineering/configuration]
---

# CLI 설계

연구 코드는 대부분 명령행에서 실행된다. 원격 서버에서, 작업 스케줄러에서, 수십 개의 설정을 도는 스크립트에서 실행된다. 이때 명령행 인터페이스(command-line interface, CLI)는 코드와 실험자 사이의 계약이다. 어떤 인자를 받는지, 기본값이 무엇인지, 잘못된 값을 주면 어떻게 되는지가 이 계약의 내용이다. 계약이 모호하면 실험자는 바꿨다고 믿은 값이 실제로는 바뀌지 않은 실험을 돌리게 된다.

이 장은 Python 표준 라이브러리 `argparse`로 subcommand를 가진 CLI를 만들고, 6.3장의 설정 파일과 명령행 인자를 일관된 우선순위로 병합하는 방법을 다룬다. 목표는 동료가 문서 없이 `--help`만 보고 실험을 실행할 수 있는 CLI다.

## 학습 목표

- argparse로 모듈러 CLI를 만든다
- 설정 파일과 명령행 인자를 일관되게 병합한다

## argparse

**왜 필요한가.** `sys.argv`를 직접 잘라 쓰는 스크립트는 인자의 순서를 외워야 하고, 타입 변환을 곳곳에서 하고, 잘못된 입력에 알아보기 어려운 오류를 낸다. `argparse`는 인자의 이름·타입·기본값·설명을 한곳에 선언하면 해석, 타입 변환, 오류 메시지, 도움말을 만들어 준다.

**정의.** `argparse`가 다루는 인자는 두 종류다.

- **위치 인자**(positional argument): 이름 없이 순서로 정해진다. 반드시 필요한 입력 하나(설정 파일, checkpoint 경로)에 쓴다.
- **옵션**(optional argument): `--lr 3e-4`처럼 이름으로 주는 인자다. 주지 않으면 기본값을 쓴다.

각 인자에 `type`(문자열을 변환할 함수), `default`, `choices`(허용 값 목록), `help`(설명), `action`(예: 값 없이 켜는 `store_true`, 여러 번 받는 `append`)을 지정한다. 아래 셀은 `error` 메서드를 덮어써서, 원래는 사용법을 출력하고 프로그램을 끝내는 오류를 예외로 바꿔 확인한다(오류 메시지의 문구는 Python 버전에 따라 조금 다를 수 있다. 집필 시점 Python 3.13).

```python {exec="browser" session="cli"}
import argparse

class Parser(argparse.ArgumentParser):
    def error(self, message):            # 기본 동작(메시지 출력 후 종료)을 예외로 바꿔 셀에서 확인한다
        raise ValueError(message)

p = Parser(prog="train", description="모델을 학습한다.")
p.add_argument("config", help="설정 파일 경로")                    # 위치 인자
p.add_argument("--lr", type=float, help="학습률")
p.add_argument("--epochs", type=int, default=50, help="epoch 수")
p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
p.add_argument("--dry-run", action="store_true", help="설정만 출력하고 끝낸다")

print(p.parse_args(["configs/base.toml", "--lr", "3e-4", "--dry-run"]))
for argv in [["configs/base.toml", "--lr", "fast"], ["configs/base.toml", "--device", "tpu"], []]:
    try:
        p.parse_args(argv)
    except ValueError as e:
        print("오류:", e)
```

```output
Namespace(config='configs/base.toml', lr=0.0003, epochs=50, device='cpu', dry_run=True)
오류: argument --lr: invalid float value: 'fast'
오류: argument --device: invalid choice: 'tpu' (choose from 'cpu', 'cuda')
오류: the following arguments are required: config
```

해석 결과는 `Namespace` 객체다. `--lr`의 문자열 `"3e-4"`는 실수 0.0003으로, `--dry-run`은 `dry_run=True`로 바뀌었다(옵션 이름의 하이픈은 속성 이름에서 밑줄이 된다). 타입이 틀린 값, 허용되지 않은 값, 빠진 위치 인자는 계산을 시작하기 전에 원인을 가리키는 메시지로 거부된다. 실제 프로그램에서는 이 메시지와 사용법을 출력하고 종료 코드 2로 끝난다.

**도움말.** `-h`/`--help`는 자동으로 생긴다. `help` 문자열에 `%(default)s`를 쓰면 기본값이 도움말에 나타나고, `formatter_class=argparse.ArgumentDefaultsHelpFormatter`를 주면 모든 옵션에 기본값이 붙는다. 도움말은 사용자가 가장 먼저, 때로는 유일하게 읽는 문서다. 각 옵션의 `help`에 단위와 허용 범위를 쓴다.

**조용히 일어나는 일.** argparse에는 연구용 CLI에서 문제가 되는 기본 동작이 두 가지 있다. 첫째, `allow_abbrev`가 기본으로 켜져 있어 옵션 이름의 **고유한 접두사**를 그 옵션으로 받아들인다. 둘째, `parse_known_args`는 모르는 인자를 오류 없이 따로 모아 돌려준다. 그 목록을 버리면 오타가 사라진다.

```python {exec="browser" session="cli"}
loose = argparse.ArgumentParser()
loose.add_argument("--epochs", type=int, default=50)
loose.add_argument("--lr-schedule", default="cosine")

argv = ["--epoch", "5", "--lr", "3e-4", "--learning-rate", "3e-4"]
args, unknown = loose.parse_known_args(argv)
print("해석:", vars(args), " 버려짐:", unknown)

strict = Parser(allow_abbrev=False)
strict.add_argument("--epochs", type=int, default=50)
strict.add_argument("--lr-schedule", default="cosine")
try:
    strict.parse_args(argv)
except ValueError as e:
    print("오류:", e)
```

```output
해석: {'epochs': 5, 'lr_schedule': '3e-4'}  버려짐: ['--learning-rate', '3e-4']
오류: unrecognized arguments: --epoch 5 --lr 3e-4 --learning-rate 3e-4
```

첫 parser에서 `--epoch`는 `--epochs`로 받아들여졌다. 여기까지는 편의 기능이다. 그런데 학습률을 바꾸려던 `--lr 3e-4`는 `--lr-schedule`의 접두사라서 **학습률 스케줄을 `'3e-4'`로** 바꿨고, `--learning-rate`는 버려졌다. 오류는 하나도 없다. 두 번째 parser처럼 `allow_abbrev=False`와 `parse_args`를 쓰면 세 인자 모두 거부된다. 연구용 CLI의 원칙은 단순하다. **모르는 입력은 즉시 실패한다.**

::quiz{id="abbrev-prefix"}

## subcommand

**왜 필요한가.** 프로젝트에는 학습, 평가, 예측, 모델 내보내기처럼 여러 작업이 있다. 작업마다 스크립트를 따로 두면 공통 옵션(seed, 설정 파일, 로그 수준)의 이름과 의미가 스크립트마다 조금씩 달라진다. **subcommand**는 `git commit`, `git log`처럼 하나의 명령 아래 작업을 나누는 방식이다. `soh train`, `soh eval`, `soh predict`, `soh export`가 한 CLI의 subcommand가 되면, 공통 옵션은 한곳에서 정의되고 사용자는 하나의 `--help`에서 출발한다.

**구현.** `add_subparsers`로 subcommand를 등록하고, 각 subcommand parser에 자기 인자를 붙인다. 여러 subcommand가 공유하는 옵션은 `add_help=False`인 parser에 모아 `parents=[...]`로 넘긴다. `set_defaults(func=...)`로 subcommand마다 실행할 함수를 연결하면 분기문 없이 `args.func(args)`로 실행된다.

```python {exec="browser" session="cli"}
def cmd_train(args):
    return f"train: config={args.config} seed={args.seed}"

def cmd_eval(args):
    return f"eval: ckpt={args.ckpt} split={args.split} seed={args.seed}"

common = argparse.ArgumentParser(add_help=False)       # 여러 subcommand가 공유하는 옵션
common.add_argument("--seed", type=int, default=0)

cli = Parser(prog="soh")
sub = cli.add_subparsers(dest="command", required=True)
t = sub.add_parser("train", parents=[common], help="모델을 학습한다")
t.add_argument("config")
t.set_defaults(func=cmd_train)
e = sub.add_parser("eval", parents=[common], help="checkpoint를 평가한다")
e.add_argument("ckpt")
e.add_argument("--split", choices=["val", "test"], default="test")
e.set_defaults(func=cmd_eval)

for argv in [["train", "configs/base.toml", "--seed", "1"], ["eval", "outputs/run1/model.npz"], ["predict"]]:
    try:
        args = cli.parse_args(argv)
        print(args.func(args))
    except ValueError as err:
        print("오류:", err)
```

```output
train: config=configs/base.toml seed=1
eval: ckpt=outputs/run1/model.npz split=test seed=0
오류: argument command: invalid choice: 'predict' (choose from 'train', 'eval')
```

`--seed`는 두 subcommand에서 같은 이름, 같은 타입, 같은 기본값을 가진다. 등록되지 않은 subcommand는 사용 가능한 목록과 함께 거부된다. `required=True`를 빠뜨리면 subcommand 없이 실행했을 때 `args.func`가 없어 엉뚱한 곳에서 `AttributeError`가 난다.

공유 옵션을 최상위 parser에 정의하는 방법도 있지만, 그러면 그 옵션은 subcommand 이름 **앞**에 와야 한다(`soh --seed 1 train ...`). 사용자는 대개 `soh train ... --seed 1`처럼 쓰므로, `parents`로 각 subcommand에 넣는 편이 덜 놀랍다.

::quiz{id="subcommand-option-position"}

**모듈 배치.** CLI 코드는 `src/<project>/cli.py`에 두고, 그 안에는 parser를 만드는 함수와 각 subcommand를 실행하는 얇은 함수만 둔다. 실제 학습·평가 로직은 6.1장의 책임별 모듈(`training/`, `evaluation/`)에 있다. `main(argv=None)`처럼 인자 목록을 받는 함수로 만들면, 테스트에서 `main(["train", "--dry-run"])`처럼 셸 없이 호출할 수 있다.

## 다른 도구

`argparse` 말고도 Python CLI를 만드는 서드파티 라이브러리가 있다(집필 시점 예시. 기능과 사용법은 각 도구의 문서에서 확인한다).

| 도구 | 방식 | 맞는 경우 |
|---|---|---|
| argparse | parser 객체에 인자를 선언 | 의존성 없이, 동작을 세밀하게 제어할 때 |
| click | 함수에 데코레이터로 옵션을 선언 | 명령이 많고 중첩된 큰 CLI |
| typer | 함수의 type hint에서 옵션을 만듦(click 기반) | 함수 시그니처가 곧 CLI인 간결한 도구 |

어느 도구를 쓰든 이 장의 원칙은 같다. 모르는 입력은 즉시 실패하고, 기본값은 한곳에 두고 도움말에 보여 주고, 설정 파일과의 우선순위를 일관되게 지킨다. 도구를 고를 때는 이 원칙을 그 도구에서 어떻게 지키는지(예: 모르는 옵션의 처리, 기본값이 "주지 않음"과 구분되는지)를 먼저 확인한다. 6.3장의 Hydra처럼 설정 관리 도구가 명령행 override 문법까지 제공하는 경우도 있다. 그때는 CLI와 설정의 경계를 그 도구가 정한 방식에 맞춘다.

## 설정과 CLI의 병합

**우선순위.** 6.3장의 설정 층에 명령행이 마지막 층으로 올라간다. 우선순위는 **기본값 < 설정 파일 < 명령행**이고, 오른쪽일수록 우선한다. 기본 설정은 모든 실험의 출발점이고, 설정 파일은 한 실험을, 명령행은 이번 실행 한 번을 나타낸다. 더 좁은 범위의 의도가 더 넓은 범위의 의도를 이긴다.

**흔한 버그.** 이 우선순위를 깨는 가장 흔한 방법은 parser에도 기본값을 적는 것이다. 명령행에서 주지 않은 옵션도 `Namespace`에는 parser 기본값으로 들어 있으므로, 마지막에 병합하면 설정 파일의 값을 덮어쓴다.

```python {exec="browser" session="cli"}
defaults = {"lr": 1e-3, "epochs": 50, "seed": 0}
file_cfg = {"lr": 3e-4, "epochs": 100}                 # 설정 파일에서 읽었다고 하자

bad = argparse.ArgumentParser()
bad.add_argument("--lr", type=float, default=1e-3)     # 기본값을 parser에도 적었다
bad.add_argument("--epochs", type=int, default=50)
good = argparse.ArgumentParser()
good.add_argument("--lr", type=float, default=None)    # 주지 않은 인자는 None
good.add_argument("--epochs", type=int, default=None)

argv = ["--epochs", "20"]
for name, parser in [("bad ", bad), ("good", good)]:
    cli_args = {k: v for k, v in vars(parser.parse_args(argv)).items() if v is not None}
    print(name, {**defaults, **file_cfg, **cli_args})
```

```output
bad  {'lr': 0.001, 'epochs': 20, 'seed': 0}
good {'lr': 0.0003, 'epochs': 20, 'seed': 0}
```

두 parser 모두 명령행의 `--epochs 20`은 반영했다. 그러나 `bad`는 사용자가 건드리지 않은 `lr`을 parser 기본값 0.001로 되돌려, 설정 파일의 3e-4를 지웠다. `good`은 옵션의 parser 기본값을 `None`으로 두고, `None`이 아닌 값만 병합한다. `default=argparse.SUPPRESS`를 주어 주지 않은 옵션이 `Namespace`에 아예 나타나지 않게 하는 방법도 있다.

::quiz{id="parser-default-overrides-file"}

**이름을 하나로.** 명령행 옵션과 설정 키의 이름이 다르면(`--learning-rate`와 `train.lr`, `--bs`와 `train.batch_size`), 사용자는 두 이름의 대응을 외워야 하고, 저장된 설정에서 명령행을 거꾸로 찾기 어렵다. 가장 단순한 해결은 옵션의 `dest`를 설정 키의 점 경로로 두는 것이다.

```python
g.add_argument("--lr", dest="train.lr", type=float, default=None, help="학습률 (기본값: 0.1)")
```

이렇게 하면 `vars(args)`가 곧 `{"train.lr": 0.0003, ...}` 형태의 점 경로 dict가 되어 6.3장의 override와 같은 방식으로 병합된다. 도움말의 metavar도 `--lr TRAIN.LR`로 표시되어 어떤 설정 키를 바꾸는지 드러난다. 자주 바꾸는 몇 개의 키만 전용 옵션을 주고, 나머지는 `--set data.noise=0.2` 같은 범용 override로 받는다.

**기본값을 보여 준다.** parser 기본값을 `None`으로 두면 `ArgumentDefaultsHelpFormatter`는 도움말에 `(default: None)`을 표시한다. 사용자에게는 쓸모없는 정보다. 실제 기본값은 기본 설정에 있으므로, 도움말 문자열을 만들 때 기본 설정에서 값을 가져와 넣는다. 그리고 `--dry-run` 같은 옵션으로 **최종 설정과 각 값이 어느 층에서 왔는지**를 출력하게 하면, 실행 전에 의도한 값이 들어갔는지 확인할 수 있다.

:::note[분야 예시]
기상 관측소 자료로 예측 모델을 돌리는 연구자가 작업 스케줄러에 같은 명령을 지역별로 수십 번 제출한다고 하자. `--region`, `--seed`만 바꾸고 나머지는 설정 파일을 따르는 것이 의도다. parser 기본값이 설정 파일을 덮어쓰는 버그가 있으면, 설정 파일에서 바꾼 입력 변수 목록이 모든 작업에서 조용히 기본값으로 돌아간다. 작업마다 `--dry-run` 출력을 로그 앞부분에 남기면, 이런 차이를 결과가 나오기 전에 찾을 수 있다.
:::

## 진입점 등록

`python src/my_project/cli.py train ...`처럼 파일 경로로 실행하면 작업 디렉터리에 따라 import가 깨지기 쉽다(5.2장). 설치한 패키지의 CLI를 실행하는 방법은 두 가지다.

**`python -m`.** 패키지에 `__main__.py`를 두면 `python -m my_project`로 실행할 수 있다. 모듈 하나를 실행하려면 `python -m my_project.cli`처럼 모듈 경로를 쓴다.

```python
# src/my_project/__main__.py
from my_project.cli import main

main()
```

**console script.** `pyproject.toml`의 `[project.scripts]`에 명령 이름과 호출할 함수를 적으면, 패키지를 설치할 때 그 이름의 실행 파일이 만들어진다.

```toml
[project.scripts]
soh = "soh_tools.cli:main"
```

이 실행 파일은 `soh_tools.cli`를 import하고 `main()`을 호출한다. 모듈은 import되므로 `if __name__ == "__main__":` 블록은 실행되지 않는다. 그래서 `main()`은 인자 없이 호출되어도 `sys.argv`에서 명령행을 읽어야 한다(`parser.parse_args(None)`이 그렇게 한다). 실습의 `soh_cli.py`를 `src/soh_tools/cli.py`로 옮긴 작은 패키지에 위 설정과 `__main__.py`를 넣고, 집필 시점에 `uv run soh --help`와 `uv run python -m soh_tools train --dry-run`이 모두 동작하는 것을 확인했다. 패키지 빌드와 설치 명령은 3부에서 다룬 도구의 문서에서 확인한다.

## 실습

프로젝트에 train/eval subcommand를 가진 CLI를 만들고 `--help`만으로 사용법을 알 수 있게 한다. 예시 구현은 `labs/research-code-engineering/cli-design/soh_cli.py`다. 합성 데이터로 회귀 모델을 학습·평가하며, 이 장의 원칙을 모두 적용했다.

- `allow_abbrev=False`와 `parse_args`로 모르는 옵션은 즉시 실패한다.
- 설정 override 옵션은 `dest`가 설정 키의 점 경로이고 parser 기본값은 `None`이다. 도움말에는 기본 설정의 실제 값을 보인다.
- 기본값 < `--config` 파일 < 개별 옵션·`--set` 순서로 병합하고, 모르는 설정 키와 타입 불일치는 해당 subcommand의 사용법과 함께 오류로 끝난다(종료 코드 2).
- `train --dry-run`은 최종 설정과 각 값의 출처를 출력한다.

다음은 집필 시점(Python 3.13)의 출력이다. 도움말의 줄바꿈과 배치는 Python 버전과 터미널 폭에 따라 조금 다를 수 있다.

```bash
cd labs/research-code-engineering/cli-design
uv run --project ../.. python soh_cli.py --help
```

```output
usage: soh_cli [-h] COMMAND ...

합성 데이터로 회귀 모델을 학습·평가한다. 설정 우선순위: 기본값 < --config < 개별 옵션·--set

positional arguments:
  COMMAND
    train     모델을 학습하고 run 디렉터리에 저장한다
    eval      저장된 run을 평가한다

options:
  -h, --help  show this help message and exit

명령별 옵션은 soh_cli COMMAND --help로 본다.
```

```bash
uv run --project ../.. python soh_cli.py train --help
```

```output
usage: soh_cli train [-h] [--config CONFIG] [--out OUT] [--dry-run]
                     [--data {sine,linear}] [--model {poly,linear}]
                     [--degree MODEL.DEGREE] [--lr TRAIN.LR]
                     [--epochs TRAIN.EPOCHS] [--seed TRAIN.SEED]
                     [--set KEY=VALUE]

모델을 학습하고 config.json·model.npz를 --out 디렉터리에 저장한다.

options:
  -h, --help            show this help message and exit
  --config CONFIG       TOML 설정 파일 (없으면 기본값만 사용)
  --out OUT             결과를 저장할 새 디렉터리 (기본값: outputs/<시작 시각>)
  --dry-run             최종 설정과 각 값의 출처만 출력하고 끝낸다

설정 override (주지 않으면 설정 파일 또는 기본값):
  --data {sine,linear}  데이터셋 (기본값: sine)
  --model {poly,linear}
                        모델 (기본값: poly)
  --degree MODEL.DEGREE
                        다항식 차수 (기본값: 5)
  --lr TRAIN.LR         학습률 (기본값: 0.1)
  --epochs TRAIN.EPOCHS
                        epoch 수 (기본값: 2000)
  --seed TRAIN.SEED     난수 seed (기본값: 0)
  --set KEY=VALUE       그 밖의 설정 키 덮어쓰기, 예: --set data.noise=0.2 (여러 번 가능)
```

설정 파일 `configs/linear.toml`은 모델을 선형으로, 학습률을 0.05로, seed를 3으로 바꾼다. 명령행에서 seed만 1로 다시 바꾸고 `--dry-run`으로 확인한다.

```bash
uv run --project ../.. python soh_cli.py train --config configs/linear.toml --seed 1 --dry-run
```

```output
data.n_train   = 200      (default)
data.name      = 'sine'   (default)
data.noise     = 0.1      (default)
model.degree   = 5        (default)
model.name     = 'linear' (file)
train.epochs   = 2000     (default)
train.lr       = 0.05     (file)
train.seed     = 1        (cli)
```

설정 파일의 seed 3은 명령행의 1에 졌고, 명령행에서 주지 않은 학습률은 설정 파일의 0.05가 유지되었다. 학습하고 평가한다. 평가는 run 디렉터리의 `config.json`과 `model.npz`만 읽는다(6.1장).

```bash
uv run --project ../.. python soh_cli.py train --config configs/linear.toml --seed 1 --out outputs/linear-s1
uv run --project ../.. python soh_cli.py eval outputs/linear-s1 --split val
```

```output
학습 완료: outputs/linear-s1/model.npz (train RMSE 0.4512)
val RMSE = 0.4538  (outputs/linear-s1)
```

잘못된 입력은 실행 전에 거부된다. 다음은 세 명령의 오류 줄이다(앞에 사용법이 함께 출력된다).

```bash
uv run --project ../.. python soh_cli.py train --set train.learning_rate=0.3
uv run --project ../.. python soh_cli.py train --set train.epochs=1e3
uv run --project ../.. python soh_cli.py train --epoch 5
```

```output
soh_cli train: error: 알 수 없는 설정 키 'train.learning_rate' (cli)
soh_cli train: error: train.epochs: int여야 합니다. 받은 값 1000.0 (cli)
soh_cli: error: unrecognized arguments: --epoch 5
```

이어서 자신의 프로젝트에 같은 구조의 CLI를 만든다.

1. `src/<project>/cli.py`에 `build_parser()`와 `main(argv=None)`을 두고, train·eval subcommand를 6.1장의 학습·평가 함수에 연결한다.
2. 자주 바꾸는 설정 키 몇 개에 전용 옵션(`dest`=점 경로, 기본값 `None`)을, 나머지에 `--set`을 둔다. 6.3장의 설정 객체 검증을 병합 뒤에 붙인다.
3. `[project.scripts]`에 명령을 등록하고, 설치한 뒤 다른 디렉터리에서 `<명령> --help`와 `<명령> train --dry-run`을 실행한다.
4. 동료에게 `--help` 출력만 주고 특정 설정의 실험을 실행해 달라고 부탁한다. 질문을 받은 곳은 도움말이 부족한 곳이다.

마지막으로 세 층을 병합하며 값의 출처를 기록하는 함수를 연습문제로 구현한다.

::exercise{id="cli-config-merge"}

## 실패 사례

- **인자 이름과 설정 키가 달라 혼란을 주는 것.** `--learning-rate`는 `train.lr`을, `--bs`는 `train.batch_size`를 바꾼다는 대응을 사용자가 외워야 한다. 대응표가 어긋나면 옵션이 엉뚱한 키를 바꾸거나 아무 키도 바꾸지 않는다. 저장된 설정에서 그 실행의 명령행을 거꾸로 만들기도 어렵다.
- **잘못된 인자를 조용히 무시하는 것.** `parse_known_args`의 나머지를 버리거나, 접두사 일치를 켜 두거나, 설정 병합이 모르는 키를 새 키로 추가한다. 실험자는 바꿨다고 믿은 값으로 결과를 해석한다.
- **기본값을 숨기는 것.** 도움말에 기본값이 없거나 `(default: None)`만 보이면, 사용자는 아무것도 주지 않았을 때 어떤 값으로 실행되는지 코드를 읽어야 안다. 기본값이 parser, 설정 파일, 코드 속 `.get()` 세 곳에 흩어지면 코드를 읽어도 모른다.
- **parser 기본값이 설정 파일을 덮어쓰는 것.** 명령행에서 주지 않은 옵션이 parser 기본값으로 마지막 층에 들어가 설정 파일의 값을 지운다. 오류 없이 실행된다.
- **실패해도 종료 코드 0으로 끝나는 것.** 오류 메시지만 출력하고 정상 종료하면, 스케줄러와 자동화 스크립트는 실패한 실험을 성공으로 기록한다. 실패는 0이 아닌 종료 코드로 알린다(오류 처리는 6.5장).

## 통과 기준

- 동료가 문서 없이 `--help`만 보고 지정한 설정(데이터셋·모델·seed·학습률)의 학습과 평가를 실행한다.
- `--dry-run` 출력에서 각 설정 값의 출처(기본값·설정 파일·명령행)를 확인할 수 있고, 명령행에서 주지 않은 옵션이 설정 파일 값을 바꾸지 않는다.
- 모르는 옵션·설정 키와 타입이 틀린 값이 0이 아닌 종료 코드로 거부된다.
- 연습문제 `cli-config-merge`의 테스트를 통과한다.
