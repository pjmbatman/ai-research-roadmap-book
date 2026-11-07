"""6.5 실습 참고 해답: print_train.py를 logging과 명시적 검증으로 바꾼 것.

- 콘솔에는 INFO 이상, run 디렉터리의 train.log에는 DEBUG까지 남긴다.
- 시작할 때 실제 설정과 데이터 통계를 남긴다.
- 읽을 수 없거나 물리적으로 불가능한 행은 행 번호와 이유를 경고로 남기고,
  건너뛴 비율이 --max-skip-frac을 넘으면 즉시 실패한다(기본값 0: 하나라도 있으면 실패).
- loss가 유한하지 않으면 그 epoch에서 멈춘다.

실행:
  uv run --project labs python labs/research-code-engineering/logging-and-errors/logged_train.py --run-dir outputs/run-001
  uv run --project labs python labs/research-code-engineering/logging-and-errors/logged_train.py --run-dir outputs/run-002 --max-skip-frac 0.4
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path

from print_train import ROWS

log = logging.getLogger("lab.train")


class DataValidationError(ValueError):
    """입력 데이터가 실험의 가정을 어긴다."""


def setup_logging(run_dir: Path, console_level: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("lab")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    to_file = logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8")
    to_file.setLevel(logging.DEBUG)
    to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s"))
    root.addHandler(console)
    root.addHandler(to_file)


def load(rows: list[str], max_skip_frac: float) -> list[tuple[float, float]]:
    data, problems = [], []
    for lineno, line in enumerate(rows[1:], start=2):           # 머리글이 1행
        try:
            t, y = line.split(",")
            temp_k, value = float(t), float(y)
        except ValueError as err:
            problems.append((lineno, f"읽을 수 없음: {line!r} ({err})"))
            continue
        if not (math.isfinite(temp_k) and temp_k > 0):
            problems.append((lineno, f"temp_k={temp_k}: 절대온도는 양수여야 함"))
            continue
        data.append((temp_k, value))
    for lineno, reason in problems:
        log.warning("행 %d 건너뜀: %s", lineno, reason)
    frac = len(problems) / max(len(rows) - 1, 1)
    if frac > max_skip_frac:
        raise DataValidationError(f"{len(rows) - 1}행 중 {len(problems)}행이 잘못됨 "
                                  f"({frac:.0%} > 허용 {max_skip_frac:.0%})")
    temps = [t for t, _ in data]
    log.info("데이터 %d행 (건너뜀 %d), temp_k 범위 [%.1f, %.1f]", len(data), len(problems), min(temps), max(temps))
    return data


def train(data: list[tuple[float, float]], lr: float, epochs: int) -> float:
    w = 0.0
    for epoch in range(epochs):
        loss = 0.0
        for temp_k, y in data:
            err = w * temp_k - y
            loss += err**2
            w -= lr * 2 * err * temp_k
        loss /= len(data)
        if not math.isfinite(loss):
            raise FloatingPointError(f"epoch {epoch}: loss={loss} (lr={lr})")
        log.info("epoch %d/%d loss=%.6g", epoch + 1, epochs, loss)
        log.debug("epoch %d w=%.6g", epoch + 1, w)
    return w


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-skip-frac", type=float, default=0.0)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = p.parse_args(argv)
    setup_logging(args.run_dir, args.log_level)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    log.info("config %s", json.dumps(config, ensure_ascii=False))
    try:
        data = load(ROWS, args.max_skip_frac)
        w = train(data, args.lr, args.epochs)
    except (DataValidationError, FloatingPointError) as e:     # 예상한 실패: 짧은 메시지와 종료 코드
        log.error("%s: %s", type(e).__name__, e)
        return 2
    log.info("w = %.6g", w)
    (args.run_dir / "result.json").write_text(json.dumps({"w": w}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log.exception("예상하지 못한 오류")                       # 예상하지 못한 실패: traceback을 로그에도
        raise
