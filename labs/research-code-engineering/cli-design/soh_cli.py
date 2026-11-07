"""6.4 실습: train/eval subcommand를 가진 연구용 CLI.

실행 예 (이 디렉터리에서):
  uv run --project ../.. python soh_cli.py --help
  uv run --project ../.. python soh_cli.py train --help
  uv run --project ../.. python soh_cli.py train --model linear --seed 1 --out outputs/linear-s1
  uv run --project ../.. python soh_cli.py eval outputs/linear-s1 --split val

설계 원칙
- 명령행 옵션의 dest를 설정 키와 같은 점 경로("train.lr")로 둔다. 이름이 하나뿐이다.
- 옵션의 parser 기본값은 None이다. 주지 않은 옵션은 설정 파일 값을 덮어쓰지 않는다.
- help에는 실제 기본값(DEFAULTS)을 보여 준다. 기본값을 숨기지 않는다.
- 모르는 옵션·설정 키는 즉시 오류로 끝낸다(parse_args, allow_abbrev=False).
"""
import argparse
import copy
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULTS = {
    "data": {"name": "sine", "n_train": 200, "noise": 0.1},
    "model": {"name": "poly", "degree": 5},
    "train": {"lr": 0.1, "epochs": 2000, "seed": 0},
}


# ---------------------------------------------------------------- 설정 병합
def set_dotted(cfg: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for key in parents:
        node = node[key]
    node[leaf] = value


def resolve(file_cfg: dict, cli_args: dict) -> tuple[dict, dict]:
    """우선순위 DEFAULTS < 설정 파일 < 명령행. 각 키의 출처도 함께 반환한다."""
    known = _flatten(DEFAULTS)
    cfg, sources = copy.deepcopy(DEFAULTS), dict.fromkeys(known, "default")
    for source, flat in [("file", _flatten(file_cfg)), ("cli", cli_args)]:
        for dotted, value in flat.items():
            if dotted not in known:
                raise ValueError(f"알 수 없는 설정 키 {dotted!r} ({source})")
            expected = type(known[dotted])
            if expected is float and type(value) is int:
                value = float(value)
            if type(value) is not expected:
                raise ValueError(f"{dotted}: {expected.__name__}여야 합니다. 받은 값 {value!r} ({source})")
            set_dotted(cfg, dotted, value)
            sources[dotted] = source
    return cfg, sources


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        flat.update(_flatten(v, path) if isinstance(v, dict) else {path: v})
    return flat


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        dotted, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set은 KEY=VALUE 형식이어야 합니다: {item!r}")
        try:
            out[dotted] = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            out[dotted] = raw
    return out


# ---------------------------------------------------------------- 실험
def make_data(data: dict, n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)                                     # (N,)
    clean = np.sin(np.pi * x) if data["name"] == "sine" else 2.0 * x - 0.5  # (N,)
    return x, clean + data["noise"] * rng.normal(size=n)


def features(model: dict, x):
    degree = model["degree"] if model["name"] == "poly" else 1
    return np.stack([x**k for k in range(degree + 1)], axis=1)            # (N, P)


SPLIT_SEED_OFFSET = {"val": 1_000, "test": 10_000}


def cmd_train(cfg: dict, args) -> None:
    x, y = make_data(cfg["data"], cfg["data"]["n_train"], cfg["train"]["seed"])
    X = features(cfg["model"], x)                                           # (N, P)
    w = np.zeros(X.shape[1])                                                # (P,)
    for _ in range(cfg["train"]["epochs"]):
        w -= cfg["train"]["lr"] * 2.0 * X.T @ (X @ w - y) / len(y)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    np.savez(args.out / "model.npz", w=w)
    print(f"학습 완료: {args.out}/model.npz (train RMSE {np.sqrt(np.mean((X @ w - y) ** 2)):.4f})")


def cmd_eval(cfg: dict, args) -> None:
    saved = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    w = np.load(args.run_dir / "model.npz")["w"]                           # (P,)
    seed = saved["train"]["seed"] + SPLIT_SEED_OFFSET[args.split]
    x, y = make_data(saved["data"], 1000, seed)
    rmse = np.sqrt(np.mean((features(saved["model"], x) @ w - y) ** 2))
    print(f"{args.split} RMSE = {rmse:.4f}  ({args.run_dir})")


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soh_cli", allow_abbrev=False,
        description="합성 데이터로 회귀 모델을 학습·평가한다. 설정 우선순위: 기본값 < --config < 개별 옵션·--set",
        epilog="명령별 옵션은 soh_cli COMMAND --help로 본다.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    train = sub.add_parser("train", allow_abbrev=False, help="모델을 학습하고 run 디렉터리에 저장한다",
                           description="모델을 학습하고 config.json·model.npz를 --out 디렉터리에 저장한다.")
    train.add_argument("--config", type=Path, help="TOML 설정 파일 (없으면 기본값만 사용)")
    train.add_argument("--out", type=Path, help="결과를 저장할 새 디렉터리 (기본값: outputs/<시작 시각>)")
    train.add_argument("--dry-run", action="store_true", help="최종 설정과 각 값의 출처만 출력하고 끝낸다")
    g = train.add_argument_group("설정 override (주지 않으면 설정 파일 또는 기본값)")
    D = DEFAULTS
    g.add_argument("--data", dest="data.name", choices=["sine", "linear"], help=f"데이터셋 (기본값: {D['data']['name']})")
    g.add_argument("--model", dest="model.name", choices=["poly", "linear"], help=f"모델 (기본값: {D['model']['name']})")
    g.add_argument("--degree", dest="model.degree", type=int, help=f"다항식 차수 (기본값: {D['model']['degree']})")
    g.add_argument("--lr", dest="train.lr", type=float, help=f"학습률 (기본값: {D['train']['lr']})")
    g.add_argument("--epochs", dest="train.epochs", type=int, help=f"epoch 수 (기본값: {D['train']['epochs']})")
    g.add_argument("--seed", dest="train.seed", type=int, help=f"난수 seed (기본값: {D['train']['seed']})")
    g.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                   help="그 밖의 설정 키 덮어쓰기, 예: --set data.noise=0.2 (여러 번 가능)")
    train.set_defaults(func=cmd_train, sub_parser=train)

    ev = sub.add_parser("eval", allow_abbrev=False, help="저장된 run을 평가한다",
                        description="run 디렉터리의 config.json과 model.npz만으로 RMSE를 계산한다.")
    ev.add_argument("run_dir", type=Path, help="train --out으로 만든 디렉터리")
    ev.add_argument("--split", choices=["val", "test"], default="test", help="평가할 분할 (기본값: %(default)s)")
    ev.set_defaults(func=cmd_eval, sub_parser=ev)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        try:
            file_cfg = tomllib.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            cli_args = {k: v for k, v in vars(args).items() if "." in k and v is not None}
            cli_args.update(parse_set(args.sets))
            cfg, sources = resolve(file_cfg, cli_args)
        except (OSError, ValueError) as e:
            args.sub_parser.error(str(e))                                   # train 사용법과 메시지를 보이고 종료 코드 2
        if args.out is None:
            args.out = Path("outputs") / f"{datetime.now():%Y%m%d-%H%M%S}"
    else:
        cfg, sources = {}, {}
    if getattr(args, "dry_run", False):
        flat = _flatten(cfg)
        for key in sorted(flat):
            print(f"{key:14s} = {flat[key]!r:8s} ({sources[key]})")
        return
    args.func(cfg, args)


if __name__ == "__main__":
    main(sys.argv[1:])
