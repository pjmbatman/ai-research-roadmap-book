"""6.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-code-engineering/configuration/cells.py
"""
# %% cell 1 — TOML 설정 읽기
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

# %% cell 2 — dataclass로 검증하기
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

# %% cell 3 — 얕은 병합, 깊은 병합, 명령행 override
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

# %% cell 4 — 실제로 쓴 설정을 저장해야 복원된다
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
