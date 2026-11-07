"""6.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-code-engineering/logging-and-errors/cells.py
"""
# %% cell 1 — logger · handler · level
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

# %% cell 2 — JSON 줄 구조화 로그
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

# %% cell 3 — 사용자 정의 예외와 예외 체인
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

# %% cell 4 — 반복 경고와 NaN 가드
import warnings
import numpy as np

rng = np.random.default_rng(0)
batches = [rng.uniform(0.5, 2.0, size=8) for _ in range(100)]   # 100 × (8,)
for i in (13, 57, 91):
    batches[i][0] = 0.0                            # log(0) = -inf가 되는 batch 3개

def mean_log(x):
    if np.any(x <= 0):                             # NumPy의 log(0) 경고는 플랫폼에 따라 안 나므로 직접 낸다
        warnings.warn("log of non-positive value", RuntimeWarning)
    with np.errstate(divide="ignore"):
        return float(np.mean(np.log(x)))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")                # 모든 경고를 기록
    values = [mean_log(b) for b in batches]
places = {(str(w.message), w.lineno) for w in caught}   # 기본 필터는 (메시지, 위치)마다 한 번만 보인다
print(f"-inf인 batch {int(np.sum(np.isinf(values)))}개, 일어난 경고 {len(caught)}개")
print(f"기본 필터('default')라면 화면에 보이는 경고 {len(places)}개")

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
