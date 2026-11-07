"""6.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-code-engineering/project-structure/cells.py
"""
# %% cell 1 — 학습과 독립적으로 평가하기
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

# %% cell 2 — 전역 상태에 기대는 함수와 순수 함수
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

# %% cell 3 — 설정에서 run ID 만들기
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
