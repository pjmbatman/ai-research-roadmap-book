"""6.3 실습: 설정 파일 하나와 명령행 override로 실험을 실행하고, 실제로 쓴 설정(resolved config)을 저장한다.

실행 예 (저장소 루트에서):
  uv run --project labs python labs/research-code-engineering/configuration/run_experiment.py
  uv run --project labs python labs/research-code-engineering/configuration/run_experiment.py \
      --exp configs/linear-baseline.toml --set train.seed=1
  uv run --project labs python labs/research-code-engineering/configuration/run_experiment.py \
      --replay outputs/<run-id>

결과는 이 디렉터리의 outputs/<run-id>/ 아래에 쌓인다(.gitignore 대상).
"""
import argparse
import copy
import hashlib
import json
import subprocess
import tomllib
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np


def find_project_root(start: Path, marker: str = "configs/default.toml") -> Path:
    """start에서 위로 올라가며 marker 파일이 있는 디렉터리를 찾는다. 작업 디렉터리와 무관하다."""
    for d in [start, *start.parents]:
        if (d / marker).is_file():
            return d
    raise FileNotFoundError(f"{start}의 상위 어디에도 {marker}가 없습니다")


ROOT = find_project_root(Path(__file__).resolve().parent)


# ---------------------------------------------------------------- 설정 객체
@dataclass(frozen=True)
class DataConfig:
    name: str
    n_train: int
    noise: float

    def __post_init__(self):
        if self.name not in {"sine", "linear"}:
            raise ValueError(f"data.name은 'sine' 또는 'linear'여야 합니다: {self.name!r}")
        if self.n_train < 2 or self.noise < 0:
            raise ValueError("data.n_train >= 2, data.noise >= 0이어야 합니다")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    degree: int

    def __post_init__(self):
        if self.name not in {"poly", "linear"}:
            raise ValueError(f"model.name은 'poly' 또는 'linear'여야 합니다: {self.name!r}")
        if self.degree < 1:
            raise ValueError("model.degree >= 1이어야 합니다")


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    epochs: int
    seed: int

    def __post_init__(self):
        if not 0 < self.lr < 1 or self.epochs < 1:
            raise ValueError("train.lr은 (0, 1), train.epochs는 1 이상이어야 합니다")


@dataclass(frozen=True)
class PathsConfig:
    output_root: str


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    paths: PathsConfig


def build(cls, d: dict, section: str):
    """dict를 dataclass로 바꾼다. 알 수 없는 키·빠진 키·타입 불일치는 즉시 실패한다."""
    names = {f.name: f for f in fields(cls)}
    if unknown := set(d) - set(names):
        raise ValueError(f"{section}: 알 수 없는 키 {sorted(unknown)}")
    if missing := set(names) - set(d):
        raise ValueError(f"{section}: 빠진 키 {sorted(missing)}")
    values = {}
    for key, f in names.items():
        v = d[key]
        if isinstance(f.type, type) and hasattr(f.type, "__dataclass_fields__"):
            v = build(f.type, v, f"{section}.{key}" if section else key)
        elif f.type is float and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)                                   # 정수 표기(1)를 실수로 허용
        elif not isinstance(v, f.type) or isinstance(v, bool) != (f.type is bool):
            raise TypeError(f"{section}.{key}: {f.type.__name__}여야 합니다. 받은 값 {v!r}")
        values[key] = v
    return cls(**values)


# ---------------------------------------------------------------- 병합과 override
def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(item: str) -> dict:
    if "=" not in item:
        raise ValueError(f"--set은 key=value 형식이어야 합니다: {item!r}")
    dotted, raw = item.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    for key in reversed(dotted.split(".")):
        value = {key: value}
    return value


def resolve(exp: Path | None, sets: list[str]) -> Config:
    raw = tomllib.loads((ROOT / "configs/default.toml").read_text(encoding="utf-8"))
    if exp is not None:
        raw = deep_merge(raw, tomllib.loads((ROOT / exp).read_text(encoding="utf-8")))
    for item in sets:
        raw = deep_merge(raw, parse_override(item))
    return build(Config, raw, "")


# ---------------------------------------------------------------- 실험
def make_data(cfg: DataConfig, n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)                                   # (N,)
    clean = np.sin(np.pi * x) if cfg.name == "sine" else 2.0 * x - 0.5   # (N,)
    return x, clean + cfg.noise * rng.normal(size=n)


def features(cfg: ModelConfig, x):
    degree = cfg.degree if cfg.name == "poly" else 1
    return np.stack([x**k for k in range(degree + 1)], axis=1)          # (N, degree+1)


def run(cfg: Config) -> dict:
    x, y = make_data(cfg.data, cfg.data.n_train, cfg.train.seed)
    X = features(cfg.model, x)                                           # (N, P)
    w = np.zeros(X.shape[1])                                             # (P,)
    for _ in range(cfg.train.epochs):                                    # MSE에 대한 경사하강
        w -= cfg.train.lr * 2.0 * X.T @ (X @ w - y) / len(y)
    x_te, y_te = make_data(cfg.data, 1000, cfg.train.seed + 10_000)      # 독립 test set
    rmse = np.sqrt(np.mean((features(cfg.model, x_te) @ w - y_te) ** 2))
    return {"test_rmse": round(float(rmse), 6)}


def code_revision() -> dict:
    """코드 revision과 커밋되지 않은 변경 여부. Git 저장소가 아니면 None."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="설정 파일과 override로 실험을 실행한다.")
    parser.add_argument("--exp", type=Path, help="기본 설정 위에 덮어쓸 실험 파일 (root 기준)")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE", help="설정 하나 덮어쓰기, 여러 번 가능")
    parser.add_argument("--replay", type=Path, help="저장된 run 디렉터리의 설정으로 다시 실행해 metric을 비교")
    args = parser.parse_args()

    if args.replay:
        run_dir = ROOT / args.replay
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        cfg = build(Config, saved["config"], "")
        before = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        after = run(cfg)
        print(f"저장된 metric : {before}\n다시 실행     : {after}\n일치          : {before == after}")
        return

    cfg = resolve(args.exp, args.sets)
    resolved = asdict(cfg)
    canon = json.dumps({k: v for k, v in resolved.items() if k != "paths"}, sort_keys=True, separators=(",", ":"))
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{hashlib.sha256(canon.encode()).hexdigest()[:8]}"
    run_dir = ROOT / cfg.paths.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    record = {"run_id": run_id, "sources": {"exp": str(args.exp) if args.exp else None, "set": args.sets},
              "code": code_revision(), "config": resolved}
    (run_dir / "config.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = run(cfg)                                                   # 설정을 먼저 저장하고 실행한다
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    print(f"run {run_id}  model={cfg.model.name} seed={cfg.train.seed} lr={cfg.train.lr}  {metrics}")
    print(f"저장: {run_dir.relative_to(ROOT)}/config.json, metrics.json")


if __name__ == "__main__":
    main()
