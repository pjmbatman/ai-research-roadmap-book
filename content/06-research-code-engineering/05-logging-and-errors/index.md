---
id: research-code-engineering/logging-and-errors
title: "로깅과 오류 처리"
title_en: "Logging and Error Handling"
tier: core
status: draft
prerequisites: [research-code-engineering/cli-design]
---

# 로깅과 오류 처리

학습이 밤사이 끝났는데 결과가 이상하다. 이때 손에 남은 것은 터미널에 흘러간 `print` 몇 줄과 결과 파일 하나뿐인 경우가 많다. 어떤 설정으로 돌았는지, 데이터가 몇 행이었는지, loss가 언제부터 이상했는지, 중간에 무슨 경고가 났는지 알 수 없으면 원인을 찾는 대신 실험을 다시 돌려야 한다. 몇 시간짜리 실험이면 하루가 사라진다.

더 나쁜 경우는 결과가 **이상해 보이지 않을 때**다. 읽지 못한 파일을 조용히 건너뛰고, 단위가 다른 행을 그대로 쓰고, NaN이 평균에 섞여도 프로그램은 끝까지 돌고 그럴듯한 숫자를 낸다. 이 장은 두 가지를 다룬다. 나중에 원인을 찾을 수 있도록 **기록하는 법**(logging)과, 잘못된 상태에서 계속 진행하지 않도록 **실패를 드러내는 법**(오류 처리)이다.

## 학습 목표

- logging 모듈로 수준별 로그를 남긴다
- 실패를 명확하게 드러내는 코드를 작성한다

## logging

`print`는 한 가지 일만 한다. 문자열을 표준 출력에 쓴다. 연구 코드에 필요한 것은 그보다 많다. 메시지가 얼마나 중요한지(진행 상황인가, 경고인가), 어디서 나왔는지(데이터 로더인가, 학습 루프인가), 언제 나왔는지가 함께 남아야 하고, 평소에는 중요한 것만 화면에 보이되 문제가 생긴 run에서는 자세한 기록까지 읽을 수 있어야 한다. 이것을 `print`로 하려면 코드 곳곳에 `if verbose:`를 넣고 출력 대상을 손으로 바꿔야 한다. Python 표준 라이브러리의 `logging` 모듈은 이 일을 네 가지 구성 요소로 나눈다.

| 구성 요소 | 역할 | 예 |
|---|---|---|
| logger | 메시지를 만드는 이름 붙은 객체. 이름이 점으로 계층을 이룬다 | `logging.getLogger("lab.data")` |
| level | 메시지의 중요도. 정해진 수준보다 낮은 것은 버린다 | `DEBUG` < `INFO` < `WARNING` < `ERROR` < `CRITICAL` |
| handler | 메시지를 어디로 보낼지 | 콘솔(`StreamHandler`), 파일(`FileHandler`) |
| formatter | 한 줄을 어떤 모양으로 쓸지 | 시각·수준·logger 이름·메시지 |

수준은 정수다(`DEBUG` 10, `INFO` 20, `WARNING` 30, `ERROR` 40, `CRITICAL` 50). 연구 코드에서는 대체로 다음처럼 나눈다.

| 수준 | 언제 | 예 |
|---|---|---|
| `DEBUG` | 문제를 추적할 때만 필요한 자세한 정보 | batch의 shape, 중간 텐서의 통계 |
| `INFO` | 정상 진행 상황 | 설정, 데이터 크기, epoch별 loss |
| `WARNING` | 계속 진행하지만 사람이 알아야 하는 일 | 결측 행 제외, gradient clipping 발동 |
| `ERROR` | 이 작업을 완료하지 못함 | 데이터 검증 실패, checkpoint 저장 실패 |
| `CRITICAL` | 프로그램 전체를 계속할 수 없음 | 디스크 가득 참 |

메시지는 **두 번 걸러진다.** 먼저 logger의 수준이 레코드(record)를 만들지 정하고, 통과한 레코드가 각 handler로 전달되어 handler의 수준으로 다시 걸러진다. 그래서 콘솔에는 INFO 이상만, run 디렉터리의 파일에는 DEBUG까지 남기려면 logger를 DEBUG로 열어 두고 handler마다 수준을 다르게 준다.

```python {exec="browser" session="log"}
import logging
import sys
import tempfile
from pathlib import Path

run_dir = Path(tempfile.mkdtemp()) / "run-001"     # 실제로는 outputs/<run-id>/
run_dir.mkdir(parents=True)

logger = logging.getLogger("lab.train")            # 이름은 점으로 계층을 이룬다: lab → lab.train
logger.setLevel(logging.DEBUG)
logger.handlers.clear()                            # 셀을 다시 실행해도 handler가 쌓이지 않게
logger.propagate = False

console = logging.StreamHandler(sys.stdout)        # 콘솔: INFO 이상, 짧게
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
to_file = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
to_file.setLevel(logging.DEBUG)                    # 파일: DEBUG까지, logger 이름 포함
to_file.setFormatter(logging.Formatter("%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(console)
logger.addHandler(to_file)

logger.debug("batch 0: x.shape=%s", (32, 3, 64, 64))
logger.info("epoch %d/%d loss=%.4f", 1, 3, 0.8123)
logger.warning("val 세트에서 결측 %d행을 제외함", 4)
print("--- train.log ---")
print((run_dir / "train.log").read_text(encoding="utf-8"), end="")
```

```output
INFO    epoch 1/3 loss=0.8123
WARNING val 세트에서 결측 4행을 제외함
--- train.log ---
DEBUG|lab.train|batch 0: x.shape=(32, 3, 64, 64)
INFO|lab.train|epoch 1/3 loss=0.8123
WARNING|lab.train|val 세트에서 결측 4행을 제외함
```

콘솔에는 두 줄, 파일에는 세 줄이 남았다. 실제 프로젝트에서는 파일 handler의 형식에 `%(asctime)s`(시각)를 넣는다. 이 예에서 뺀 것은 출력이 실행할 때마다 달라지지 않게 하기 위해서다.

코드에서 눈여겨볼 점이 셋 있다.

- **메시지 인자는 `%` 자리표시자로 넘긴다.** `logger.debug("x.shape=%s", shape)`는 레코드가 실제로 출력될 때만 문자열을 만든다. f-string으로 쓰면 DEBUG가 꺼져 있어도 문자열을 매번 만든다. 비용이 큰 값(텐서의 통계)을 넣을 때 차이가 난다.
- **handler는 한 번만 붙인다.** notebook에서 같은 셀을 다시 실행하면 handler가 하나씩 더 붙어 같은 줄이 두 번, 세 번 찍힌다. 위 코드의 `handlers.clear()`가 그것을 막는다.
- **logger 이름의 계층.** `lab.train`의 레코드는 `propagate`가 참이면 부모 logger(`lab`, 그 위의 root)의 handler로도 전달된다. 그래서 보통은 모듈마다 `logger = logging.getLogger(__name__)`만 쓰고, handler는 최상위(`lab` 또는 root)에 한 번만 붙인다.

::quiz{id="logger-handler-levels"}

**설정은 진입점에서 한 번.** 6.1장의 구조에서 `src/<project>/` 아래의 모듈은 logger를 **얻기만** 하고, handler와 수준은 `scripts/train.py` 같은 진입점이 시작할 때 한 번 설정한다. 라이브러리 모듈이 스스로 handler를 붙이거나 `logging.basicConfig`를 호출하면, 그 모듈을 import한 모든 스크립트의 출력이 바뀐다. 콘솔 수준은 6.4장의 CLI 인자(`--log-level`)로 받고, 파일은 6.3장의 run 디렉터리(`outputs/<run-id>/train.log`)에 둔다. 그러면 결과 파일과 그 결과를 만든 실행의 기록이 한 디렉터리에 함께 있다.

**구조화 로그.** 사람이 읽는 한 줄짜리 로그는 나중에 프로그램으로 분석하기 어렵다. `loss=0.8123`을 정규식으로 뽑아내야 하고, 형식이 조금만 바뀌어도 깨진다. 기계가 읽을 기록은 한 줄에 JSON 객체 하나를 쓰는 **JSON Lines** 형식이 다루기 편하다. formatter 하나로 만들 수 있다.

```python {exec="browser" session="log"}
import io
import json
import math

class JsonLines(logging.Formatter):
    def format(self, record):
        rec = {"level": record.levelname, "event": record.getMessage()}
        rec.update(getattr(record, "fields", {}))   # extra={"fields": {...}}로 넘긴 값
        return json.dumps(rec, ensure_ascii=False)

mlog = logging.getLogger("lab.metrics")
mlog.setLevel(logging.INFO)
mlog.handlers.clear()
mlog.propagate = False
buf = io.StringIO()                                # 실제로는 run_dir / "metrics.jsonl"
handler = logging.StreamHandler(buf)
handler.setFormatter(JsonLines())
mlog.addHandler(handler)

mlog.info("config", extra={"fields": {"lr": 0.3, "seed": 0, "git": "3f2c1ab"}})
for step, loss in enumerate([2.31, 1.72, 1.40, float("inf"), float("nan")]):
    mlog.info("train", extra={"fields": {"step": step, "loss": loss}})

lines = buf.getvalue().splitlines()
print(lines[0])
print(lines[4])
records = [json.loads(s) for s in lines]
bad = [r["step"] for r in records if "loss" in r and not math.isfinite(r["loss"])]
print("처음으로 유한하지 않은 loss: step", bad[0], "/ 그때 lr =", records[0]["lr"])
```

```output
{"level": "INFO", "event": "config", "lr": 0.3, "seed": 0, "git": "3f2c1ab"}
{"level": "INFO", "event": "train", "step": 3, "loss": Infinity}
처음으로 유한하지 않은 loss: step 3 / 그때 lr = 0.3
```

로그를 다시 읽는 코드가 세 줄이다. 같은 질문("loss가 언제 처음 깨졌나")을 사람이 읽는 로그로 하려면 형식에 맞춘 파서를 따로 써야 한다. 한 가지 주의할 점이 있다. Python의 `json.dumps`는 NaN과 무한대를 `NaN`, `Infinity`로 쓰고 `json.loads`도 이를 읽지만, 이것은 JSON 표준 밖의 확장이다. 엄격한 JSON 파서(다른 언어의 라이브러리, 일부 데이터베이스)는 이 줄을 거부할 수 있다. 로그를 다른 도구로 넘길 계획이면 유한하지 않은 값을 `null`이나 문자열로 바꾸는 규칙을 정해 둔다.

실험 추적 도구(22.1장)는 이 구조화 로그를 저장·검색·시각화하는 일을 대신 해 준다. 도구를 쓰더라도 run 디렉터리에 남는 텍스트 로그와 JSON 줄 로그는 도구 서버가 없어도 읽을 수 있는 마지막 기록이다.

## 무엇을 기록하는가

로그의 목적은 **실패한 run을 다시 돌리지 않고 원인을 찾는 것**이다. 그 기준으로 보면 남길 것은 다섯 가지다.

| 무엇 | 언제 | 수준 | 예 |
|---|---|---|---|
| 설정 | 시작 | INFO | 기본값·설정 파일·명령행 override가 모두 반영된 resolved config(6.3장) |
| 환경 | 시작 | INFO | git 커밋 hash와 커밋되지 않은 변경 여부, Python·주요 패키지 버전, 장치, 호스트 이름 |
| 데이터 통계 | 데이터를 읽은 직후 | INFO | 샘플 수, 결측·제외 수, 주요 변수의 범위·평균, 클래스 비율, 분할별 크기 |
| 진행 상황 | 주기적으로 | INFO | step·epoch, loss, learning rate, 처리 속도, 경과 시간 |
| 경고 | 일어날 때 | WARNING | 건너뛴 파일과 이유, gradient clipping 횟수, 수렴하지 않은 내부 반복 |

설정과 환경은 3.6장의 환경 보고서와 같은 내용을 run마다 남기는 것이다. 몇 달 뒤 "이 결과와 저 결과는 무엇이 달랐나"라는 질문은 대부분 이 두 항목의 차이로 답할 수 있다. 데이터 통계는 조용한 데이터 문제를 잡는 가장 싼 방법이다. 학습 샘플이 어제는 48,213개였는데 오늘 41,902개라면, 코드는 같아도 결과가 같을 이유가 없다.

남기지 **않을** 것도 있다. API key·토큰·비밀번호는 로그에 남기지 않는다. 로그는 동료에게 보내지고 issue에 붙여지고 저장소에 실수로 커밋된다(4.2장). 환경변수를 통째로 덤프하면 비밀정보가 함께 남는다. 또 batch마다 텐서 전체를 INFO로 남기면 로그가 수 GB가 되어 정작 중요한 경고를 찾을 수 없다. 텐서는 shape과 요약 통계(최솟값·최댓값·NaN 개수)를, 필요할 때 DEBUG로 남긴다.

::quiz{id="what-to-log-at-start"}

:::note[분야 예시]
배터리 충방전 시험기, 분광기, 현미경처럼 장비에서 데이터를 받아 처리하는 파이프라인에서는 장비 쪽 메타데이터가 결과를 좌우한다. 채널 번호, 센서 보정(calibration) 날짜, 장비 펌웨어 버전, 측정 온도, 샘플링 주기를 데이터 통계와 함께 로그에 남겨 두면, 한 채널의 결과만 이상할 때 그 채널의 보정 기록부터 확인할 수 있다. 같은 셀을 다른 날 측정한 결과가 다를 때 차이가 시료 때문인지 장비 상태 때문인지를 가르는 근거가 된다.
:::

## 오류 처리 원칙

연구 코드의 오류 처리 원칙은 한 문장이다. **잘못된 상태로 계속 진행하지 않는다**(fail loud, fail fast). 이유는 비용의 비대칭이다. 프로그램이 즉시 멈추면 잃는 것은 그 run의 시간뿐이다. 조용히 진행하면 잘못된 결과가 표와 그림에 들어가고, 그 위에 다음 실험이 설계되고, 몇 주 뒤에야 발견된다.

실패는 두 종류로 나누어 다룬다.

- **예상한 실패**: 잘못된 설정 값, 읽을 수 없는 데이터 행, 범위를 벗어난 물리량, 없는 파일. 경계(설정을 읽을 때, 데이터를 읽을 때)에서 **명시적으로 검증**하고, 무엇이 왜 잘못됐는지 말하는 예외를 던진다.
- **예상하지 못한 실패**: 코드의 버그. 잡지 않는다. traceback 전체가 남아야 고칠 수 있다.

이 구분에서 몇 가지 규칙이 나온다.

1. **예외는 좁게 잡는다.** `except Exception:`이나 맨 `except:`는 예상한 실패와 함께 오타로 생긴 `NameError`, 잘못된 인덱스의 `IndexError`까지 삼킨다. 맨 `except:`는 Ctrl+C(`KeyboardInterrupt`)까지 잡아 프로그램을 멈추지도 못하게 한다. 잡을 예외의 종류를 적는다: `except (KeyError, ValueError):`.
2. **잡았으면 처리하거나, 맥락을 더해 다시 던진다.** 처리할 수 없는 예외를 잡아 `pass`하거나 기본값을 돌려주지 않는다. 맥락(몇 번째 행인가, 어느 파일인가)을 더해 다시 던질 때는 `raise NewError(...) from err`로 원래 예외를 원인(`__cause__`)으로 연결한다. traceback에 두 예외가 모두 남는다.
3. **의미 있는 예외 타입을 만든다.** `class DataValidationError(ValueError)`처럼 프로젝트의 실패 종류에 이름을 붙이면, 호출하는 쪽이 그 실패만 골라 처리할 수 있다. 기존 타입(`ValueError`)을 상속하면 그 타입을 잡던 코드도 그대로 동작한다.
4. **외부 입력은 `assert`가 아니라 `if ... raise`로 검증한다.** `python -O`로 실행하면 `assert` 문은 제거된다. `assert`는 "내 코드가 옳다면 항상 참"인 내부 불변조건에 쓴다.

조용한 기본값과 명시적 검증을 비교한다.

```python {exec="browser" session="log"}
class DataValidationError(ValueError):
    """입력 데이터가 실험의 가정을 어긴다."""

def read_temp_silent(row):
    try:
        return float(row["temp_k"])
    except Exception:
        return 0.0                                 # 조용한 기본값: 0 K가 데이터에 섞인다

def read_temp(row, i):
    try:
        temp_k = float(row["temp_k"])
    except (KeyError, ValueError) as err:
        raise DataValidationError(f"row {i}: temp_k를 읽을 수 없음 {row!r}") from err
    if not temp_k > 0:                             # NaN도 여기서 걸린다
        raise DataValidationError(f"row {i}: temp_k={temp_k} (양수여야 함)")
    return temp_k

rows = [{"temp_k": "293.1"}, {"temp": "301.4"}, {"temp_k": "nan"}]
print("silent:", [read_temp_silent(r) for r in rows])
for i, row in enumerate(rows):
    try:
        read_temp(row, i)
    except DataValidationError as e:
        cause = f" ← {type(e.__cause__).__name__}: {e.__cause__}" if e.__cause__ else ""
        print(f"{type(e).__name__}: {e}{cause}")
```

```output
silent: [293.1, 0.0, nan]
DataValidationError: row 1: temp_k를 읽을 수 없음 {'temp': '301.4'} ← KeyError: 'temp_k'
DataValidationError: row 2: temp_k=nan (양수여야 함)
```

조용한 버전은 두 가지 잘못을 저지른다. 열 이름이 틀린 행에 0 K라는 물리적으로 불가능한 값을 넣고, 문자열 `"nan"`은 `float`가 정상적으로 NaN으로 바꾸므로 그대로 통과시킨다. 이 리스트의 평균은 NaN이 되고, NaN이 아닌 값만 평균 내는 함수를 쓰면 0 K 때문에 평균 온도가 3분의 1로 줄어든다. 검증 버전은 두 행 모두에서 몇 번째 행이 왜 잘못됐는지 말한다. 원인 예외(`KeyError`)도 연결되어 있다.

검증 조건을 `temp_k > 0`이 아니라 `not temp_k > 0`으로 쓴 것도 의도한 것이다. NaN과의 비교는 항상 거짓이므로, `if temp_k <= 0: raise`로 쓰면 NaN이 검증을 통과한다. 거꾸로 "통과 조건이 참이 아니면 실패"로 쓰면 NaN도 걸린다. 같은 이유로 `loss == float("nan")`은 loss가 NaN이어도 거짓이다. NaN 검사는 `math.isnan`, 무한대까지 포함하면 `math.isfinite`나 `np.isfinite`를 쓴다.

::quiz{id="nan-equality"}

**NaN 가드.** 학습 중 loss나 gradient가 NaN·무한대가 되면 그 뒤의 모든 step이 의미가 없다. 매 step(또는 몇 step마다) 값이 유한한지 검사해 처음 깨진 step에서 멈추면, 그 step의 batch와 설정이 로그에 남아 원인을 좁힐 수 있다. 다음 절의 코드에 가드의 예가 있다. NumPy는 `np.errstate(divide="raise", invalid="raise")` 블록 안에서 0으로 나누기나 무효 연산을 경고 대신 `FloatingPointError`로 바꿔, 문제가 생긴 연산 자체에서 멈추게 할 수 있다. PyTorch에는 역전파에서 NaN을 만든 연산을 찾아 주는 `torch.autograd.set_detect_anomaly`가 있다. 둘 다 느려지므로 진단할 때 켠다.

다만 "유한한가"만 검사하는 가드는 발산을 늦게 잡는다. 학습률이 너무 크면 loss가 $10^{6}$, $10^{16}$, $10^{26}$으로 커지면서도 한동안 유한하다(실습의 `logged_train.py`를 `--lr 1e-4 --epochs 20`으로 실행하면 20 epoch 동안 loss가 약 $7 \times 10^{192}$까지 커지지만 끝까지 유한하다). 필요하면 "loss가 초기값의 몇 배를 넘으면 멈춘다" 같은 규칙을 함께 둔다.

**예상한 실패는 짧게, 예상하지 못한 실패는 길게.** 진입점에서는 프로젝트의 예외(`DataValidationError` 등)를 잡아 한 줄짜리 ERROR 로그를 남기고 0이 아닌 종료 코드로 끝낸다. 그 밖의 예외는 `logger.exception("...")`으로 traceback 전체를 로그 파일에 남긴 뒤 다시 던진다. `logger.exception`은 `except` 블록 안에서 ERROR 수준으로 메시지와 현재 예외의 traceback을 함께 기록한다. 종료 코드는 셸 스크립트, 작업 스케줄러, CI가 "이 작업이 실패했다"를 아는 유일한 신호이므로, 실패했는데 0으로 끝나면 안 된다(6.4장).

::quiz{id="silent-except"}

:::note[분야 예시]
분자 시뮬레이션이나 양자화학 계산의 입력 파일은 단위를 명시하지 않는 경우가 많다. 길이가 Å인지 nm인지, 에너지가 eV인지 kcal/mol인지 Hartree인지가 프로그램마다 다르다. 단위가 10배 다른 좌표를 그대로 읽으면 계산은 오류 없이 끝나지만 결과는 의미가 없다. 그래서 입력을 읽은 직후 물리적으로 가능한 범위를 검증하는 것이 효과적이다. 예를 들어 결합 길이가 0.5–3 Å 범위를 벗어나는 원자 쌍이 있으면 실패로 멈추고, 어느 분자의 어느 원자 쌍인지 메시지에 넣는다.
:::

## 경고의 관리

경고(warning)는 "아직 멈출 정도는 아니지만 알아야 하는 일"이다. Python의 `warnings` 모듈이 이를 담당하고, 경고에는 종류(category)가 있다.

| 종류 | 의미 | 예 |
|---|---|---|
| `RuntimeWarning` | 실행 중 의심스러운 계산 | NumPy의 0으로 나누기, `log(0)`, overflow |
| `UserWarning` | 라이브러리가 사용자에게 알리는 일반 경고 | 권장하지 않는 인자 조합 |
| `DeprecationWarning` | 앞으로 제거될 기능(개발자 대상) | 다음 버전에서 사라질 함수 |
| `FutureWarning` | 앞으로 동작이 바뀔 기능(사용자 대상) | 기본값이 바뀔 인자 |

경고가 화면에 보일지는 **필터**(filter)가 정한다. 필터의 동작에는 `"default"`(위치마다 한 번), `"always"`(매번), `"once"`(전체에서 한 번), `"ignore"`(무시), `"error"`(예외로 바꿈) 등이 있다. 기본 동작은 대체로 `"default"`다. 즉 **같은 코드 위치에서 난 같은 경고는 한 번만 보인다.** `DeprecationWarning`은 기본적으로 숨겨지고, `__main__`의 코드가 직접 일으킨 경우에만 보인다.

"위치마다 한 번"이 무엇을 숨기는지 확인한다. batch 100개 중 3개에 0이 섞여 `log(0) = -inf`가 된다.

```python {exec="browser" session="log"}
import warnings
import numpy as np

rng = np.random.default_rng(0)
batches = [rng.uniform(0.5, 2.0, size=8) for _ in range(100)]   # 100 × (8,)
for i in (13, 57, 91):
    batches[i][0] = 0.0                            # log(0) = -inf가 되는 batch 3개

def mean_log(x):
    return float(np.mean(np.log(x)))

for action in ["default", "always"]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(action)
        values = [mean_log(b) for b in batches]
    print(f"{action:7s}: 경고 {len(caught)}개 기록, -inf인 batch {int(np.sum(np.isinf(values)))}개")

def check_finite(name, value, step):
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name}={value} (step {step})")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")                # 경고를 꺼 두어도
    try:
        for step, b in enumerate(batches):
            check_finite("mean_log", mean_log(b), step)
    except FloatingPointError as e:
        print("가드가 멈춤:", e)                     # 가드는 첫 문제 batch에서 멈춘다
```

```output
default: 경고 1개 기록, -inf인 batch 3개
always : 경고 3개 기록, -inf인 batch 3개
가드가 멈춤: mean_log=-inf (step 13)
```

기본 필터에서는 문제 batch가 3개인데 경고는 1개다. 3개가 아니라 3,000개였어도 1개다. 그래서 로그에 경고가 한 번 찍혔다는 것은 "그 위치에서 최소 한 번 일어났다"는 뜻일 뿐이다. 경고를 모두 꺼 둔 상태에서도 값을 직접 검사하는 가드는 첫 문제 batch(step 13)에서 멈춘다. 경고는 사람에게 보내는 신호이고, 가드는 프로그램이 스스로 멈추는 장치다. 결과의 정확성이 걸린 조건은 가드로 검사한다.

::quiz{id="repeated-warning-once"}

**반복 경고를 억제하는 법과 그 위험.** 수천 줄씩 반복되는 경고는 로그를 읽을 수 없게 만든다. 그래서 스크립트 맨 위에 `warnings.filterwarnings("ignore")`를 넣고 싶어진다. 이것은 앞으로 생길 **모든** 경고, 즉 아직 보지 못한 새 문제의 신호까지 끈다. 억제는 좁게 한다.

- 종류·메시지·모듈을 지정한다: `warnings.filterwarnings("ignore", category=FutureWarning, module="some_library")`. 메시지 인자는 정규식이며 경고 메시지의 앞부분과 맞춰 본다.
- 범위를 제한한다: 꼭 필요한 블록만 `with warnings.catch_warnings(): warnings.simplefilter("ignore", ...)`로 감싼다.
- 억제한 이유를 주석에 쓴다. 라이브러리를 업그레이드할 때 그 억제가 아직 필요한지 다시 확인한다.
- 억제 대신 원인을 고칠 수 있으면 고친다. `log(0)` 경고라면 0이 들어오는 이유를 찾거나, 의도한 것이라면 `np.log(np.clip(x, eps, None))`처럼 의도를 코드로 드러낸다.

반대 방향의 도구도 있다. `python -W error script.py`나 `warnings.simplefilter("error")`는 경고를 예외로 바꾼다. 테스트에서 이 설정을 쓰면 새 경고가 생기는 순간 테스트가 실패하므로, 경고가 쌓이지 않게 관리할 수 있다(6.6장). `logging.captureWarnings(True)`를 호출하면 경고가 `py.warnings` logger를 통해 로그 handler로 전달되어, 콘솔에 흘러가는 대신 run 디렉터리의 로그 파일에 다른 기록과 함께 남는다.

:::note[분야 예시]
기상·해양 관측 자료는 결측을 `-9999`나 `1e20` 같은 채움 값(fill value)으로 표시하는 경우가 많다. 이 값을 결측으로 바꾸지 않고 평균을 내면 경고 하나 없이 결과가 틀린다. 반대로 결측을 NaN으로 바꾼 뒤 `np.mean`을 쓰면 결과가 NaN이 되고, `np.nanmean`을 쓰면 모든 값이 결측인 격자점에서 `RuntimeWarning: Mean of empty slice`가 난다. 이 경고를 일괄 억제하면 관측이 하나도 없는 지역이 NaN으로 남았다는 사실도 함께 사라진다. 결측 비율을 격자점별로 계산해 로그와 결과 파일에 함께 남기는 편이 낫다.
:::

## 실습

`print`와 조용한 `except`로 쓴 작은 학습 스크립트를 logging과 명시적 검증으로 바꾼다. 코드는 `labs/research-code-engineering/logging-and-errors/`에 있다.

- `print_train.py`(시작 코드): 온도(K)로 응답 $y$를 맞추는 1차 모델 $y \approx w \cdot T$를 SGD로 학습한다. 데이터 6행 중 한 행은 읽을 수 없고(`n/a`), 한 행은 물리적으로 불가능하다(−5.0 K).
- `logged_train.py`(참고 해답): 콘솔에는 INFO 이상, `--run-dir`의 `train.log`에는 DEBUG까지 남기고, 시작할 때 설정과 데이터 통계를 남기고, 잘못된 행은 행 번호와 이유를 경고로 남긴다. 건너뛴 비율이 `--max-skip-frac`(기본값 0)을 넘으면 종료 코드 2로 실패한다.

시작 코드부터 실행한다. 집필 환경의 출력이다.

```output
epoch 0 loss 0.7333825671398577
epoch 1 loss 0.2866994711484455
epoch 2 loss 0.1949013560750984
w = 0.003281049855009842
```

출력만 보면 아무 문제가 없다. 몇 행으로 학습했는지, 어떤 행이 빠졌는지, −5.0 K 행이 학습에 들어갔는지 알 수 없다. 참고 해답을 기본값으로 실행하면 즉시 실패한다.

```bash
uv run --project labs python labs/research-code-engineering/logging-and-errors/logged_train.py --run-dir outputs/run-001
```

```output
INFO    lab.train: config {"run_dir": "outputs/run-001", "lr": 1e-06, "epochs": 3, "max_skip_frac": 0.0, "log_level": "INFO"}
WARNING lab.train: 행 4 건너뜀: 읽을 수 없음: 'n/a,1.05' (could not convert string to float: 'n/a')
WARNING lab.train: 행 6 건너뜀: temp_k=-5.0: 절대온도는 양수여야 함
ERROR   lab.train: DataValidationError: 6행 중 2행이 잘못됨 (33% > 허용 0%)
```

건너뛰기를 명시적으로 허용하면(`--max-skip-frac 0.4`) 학습이 진행되고, 무엇을 뺐는지가 로그에 남는다.

```output
INFO    lab.train: config {"run_dir": "outputs/run-002", "lr": 1e-06, "epochs": 3, "max_skip_frac": 0.4, "log_level": "INFO"}
WARNING lab.train: 행 4 건너뜀: 읽을 수 없음: 'n/a,1.05' (could not convert string to float: 'n/a')
WARNING lab.train: 행 6 건너뜀: temp_k=-5.0: 절대온도는 양수여야 함
INFO    lab.train: 데이터 4행 (건너뜀 2), temp_k 범위 [288.0, 310.2]
INFO    lab.train: epoch 1/3 loss=0.709752
INFO    lab.train: epoch 2/3 loss=0.148049
INFO    lab.train: epoch 3/3 loss=0.0328092
INFO    lab.train: w = 0.00329334
```

마지막 epoch의 loss가 시작 코드(0.195)와 참고 해답(0.0328)에서 여섯 배쯤 다르다. 시작 코드는 −5.0 K 행을 학습과 loss 계산에 그대로 넣었기 때문이다. 기울기 $w$는 거의 같아서, 최종 파라미터만 보면 이 차이를 알아차리기 어렵다. 로그만으로 이 차이를 설명할 수 있어야 한다.

할 일은 다음과 같다.

1. `print_train.py`를 복사해 직접 logging으로 바꾼다. 참고 해답은 마지막에 비교한다.
2. 다섯 가지 기록(설정·환경·데이터 통계·진행 상황·경고)을 모두 남긴다. 참고 해답에 없는 환경 정보(git 커밋 hash, Python 버전)를 추가한다.
3. `except Exception: pass`를 없애고, 예상한 실패만 좁게 잡아 행 번호와 이유를 남긴다.
4. `--lr 1e-4 --epochs 20`으로 실행해 loss가 발산하는 것을 확인하고, 유한성 가드만으로는 잡히지 않는다는 것을 확인한 뒤 발산을 잡는 규칙을 추가한다.
5. 동료에게 `train.log` 파일만 주고, 실행 코드를 보지 않은 채 run이 왜 실패했는지 또는 무엇을 빼고 학습했는지 설명할 수 있는지 확인한다.

이어서 연습문제를 푼다. 학습이 멈춘 뒤 남은 JSON 줄 로그에서 레벨별 개수, 처음으로 loss가 깨진 step, 실행 설정, 망가진 줄의 수를 뽑는 함수를 작성한다.

::exercise{id="log-summary"}

## 실패 사례

- **`except: pass`로 오류를 숨기는 것.** 읽지 못한 파일, 변환하지 못한 값, 심지어 코드의 버그까지 흔적 없이 사라진다. 결과는 나오지만 데이터의 일부로 만든 결과다. 빠진 데이터가 특정 조건에 몰려 있으면 결론이 바뀐다. 예상한 예외만 좁게 잡고, 건너뛴 것과 이유를 기록하고, 건너뛴 비율에 한도를 둔다.
- **로그에 남지 않은 설정으로 실험하는 것.** 명령행에서 학습률을 바꿔 실행하고 그 값을 어디에도 남기지 않으면, 좋은 결과가 나와도 다시 만들 수 없다. resolved config를 run 디렉터리에 저장하고(6.3장) 로그 첫 줄에도 남긴다.
- **콘솔에만 남기는 것.** `nohup`이나 `tmux`로 띄운 학습의 콘솔 출력은 창을 닫거나 서버가 재부팅되면 사라진다. 처음부터 run 디렉터리의 파일 handler로 남긴다.
- **handler가 중복으로 붙는 것.** notebook에서 로깅 설정 셀을 여러 번 실행하거나, 여러 모듈이 각자 `basicConfig`나 `addHandler`를 호출하면 같은 줄이 두세 번씩 찍힌다. 설정은 진입점에서 한 번만 한다.
- **NaN을 `==`로 검사하는 것.** `loss == float("nan")`은 항상 거짓이라 가드가 발동하지 않는다. `math.isnan`이나 `np.isfinite`를 쓴다.
- **경고를 전역으로 끄는 것.** 스크립트 맨 위의 `warnings.filterwarnings("ignore")`는 지금 보이는 성가신 경고와 함께, 다음 달 데이터가 바뀌어 생길 새 경고도 끈다.

## 통과 기준

- 실패한 실험의 로그만으로 원인을 찾는다. 동료가 코드를 보지 않고 `train.log`만 읽어 실패 원인(또는 제외한 데이터와 그 이유)을 설명한다.
- 학습 스크립트가 시작할 때 설정·환경·데이터 통계를 run 디렉터리의 로그 파일에 남기고, 잘못된 입력에는 행·파일을 지목하는 메시지와 0이 아닌 종료 코드로 실패한다.
- 연습문제 `log-summary`의 테스트를 통과한다.

## 참고 문헌

- Python 공식 문서, *Logging HOWTO* (docs.python.org/3/howto/logging.html). — logger·handler·formatter의 구조와 레코드가 전달되는 흐름
- Python 공식 문서, *warnings — Warning control* (docs.python.org/3/library/warnings.html). — 경고 종류, 필터 동작, 기본 필터
