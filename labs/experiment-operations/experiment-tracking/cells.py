"""22.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/experiment-operations/experiment-tracking/cells.py
브라우저(Pyodide)에서는 git이 없으므로 코드 revision과 데이터 해시는 인자로 받는다.
"""
# %% cell 1 — run 기록: run ID, 메타데이터, JSON Lines metric
import hashlib
import json
import tempfile
from pathlib import Path

ROOT = Path(tempfile.mkdtemp())
IGNORED = {"out_dir", "log_level"}                 # 결과에 영향을 주지 않는 키

def config_hash(cfg):
    core = {k: v for k, v in cfg.items() if k not in IGNORED}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8]

def start_run(root, cfg, started, code_rev, data_hash):
    run_id = f"{started}-{config_hash(cfg)}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)                    # 이미 있으면 오류: 덮어쓰지 않는다
    meta = {"run_id": run_id, "config": cfg, "code_rev": code_rev, "data_hash": data_hash}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    (run_dir / "status").write_text("running")
    return run_dir

def log_metrics(run_dir, **values):                # 한 줄에 JSON 객체 하나(JSON Lines)
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(values) + "\n")

cfg = {"lr": 0.03, "seed": 0, "epochs": 5, "out_dir": "outputs"}
run_dir = start_run(ROOT / "demo", cfg, "20261107-1403", code_rev="3f2a9c1", data_hash="sha256:9e01")
log_metrics(run_dir, epoch=1, train_loss=0.912, val_loss=0.954)
log_metrics(run_dir, epoch=2, train_loss=0.611, val_loss=0.688)
print(run_dir.name, sorted(p.name for p in run_dir.iterdir()))
print((run_dir / "metrics.jsonl").read_text().splitlines()[-1])

# %% cell 2 — sweep: 설정 9개, 그중 하나는 발산, 하나는 중단
import numpy as np

def fake_train(cfg, run_dir, stop_at=None):
    """학습 대신: 설정과 seed로 정해지는 검증 손실을 epoch마다 기록한다"""
    rng = np.random.default_rng(cfg["seed"])
    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == stop_at:
            return                                 # 프로세스가 죽었다: status를 고칠 기회가 없다
        val_loss = 0.30 + np.exp(-30 * cfg["lr"] * epoch) + 0.02 * rng.standard_normal()
        if cfg["lr"] >= 0.1 and rng.random() < 0.02:
            log_metrics(run_dir, epoch=epoch, val_loss=7.9)   # 발산한 손실이 기록된다
            (run_dir / "status").write_text("failed")
            return
        log_metrics(run_dir, epoch=epoch, val_loss=round(float(val_loss), 4))
    (run_dir / "status").write_text("finished")

SWEEP = ROOT / "sweep"
for lr in [0.01, 0.03, 0.1]:
    for seed in [0, 1, 2]:
        cfg = {"lr": lr, "seed": seed, "epochs": 5, "out_dir": "outputs"}
        d = start_run(SWEEP, cfg, "20261107-1500", code_rev="3f2a9c1", data_hash="sha256:9e01")
        fake_train(cfg, d, stop_at=4 if (lr, seed) == (0.03, 2) else None)

status = [(d / "status").read_text() for d in sorted(SWEEP.iterdir())]
print({s: status.count(s) for s in sorted(set(status))})

# %% cell 3 — 결과 표: 원본 기록에서 자동으로, 끝난 run만, 빠진 수와 함께
import pandas as pd

def load_runs(root):
    rows = []
    for d in sorted(root.iterdir()):
        meta = json.loads((d / "run.json").read_text())
        last = json.loads((d / "metrics.jsonl").read_text().splitlines()[-1])
        rows.append({"run_id": meta["run_id"], "lr": meta["config"]["lr"],
                     "status": (d / "status").read_text(), "val_loss": last["val_loss"]})
    return pd.DataFrame(rows)

runs = load_runs(SWEEP)
naive = runs.groupby("lr")["val_loss"].mean()                       # 상태를 보지 않은 평균
done = runs[runs["status"] == "finished"]
table = done.groupby("lr")["val_loss"].agg(["mean", "std", "count"])
print("   lr   단순 평균 | 끝난 run 평균 ± 표준편차  n  빠짐")
for lr, r in table.iterrows():
    missing = int((runs["lr"] == lr).sum() - r["count"])
    print(f"{lr:5.2f}   {naive[lr]:8.4f} | {r['mean']:.4f} ± {r['std']:.4f}  {int(r['count'])}  {missing}")

# %% cell 4 — 표의 숫자에서 run으로, 그리고 다시 실행
best_lr = table["mean"].idxmin()
members = done[done["lr"] == best_lr]
print(f"표의 칸 lr={best_lr}: {table.loc[best_lr, 'mean']:.4f} ← run {list(members['run_id'])}")
row = members.iloc[0]
meta = json.loads((SWEEP / row["run_id"] / "run.json").read_text())
print("code_rev:", meta["code_rev"], " data:", meta["data_hash"], " config:", meta["config"])

replay = start_run(ROOT / "replay", meta["config"], "20261108-0900", meta["code_rev"], meta["data_hash"])
fake_train(meta["config"], replay)
again = json.loads((replay / "metrics.jsonl").read_text().splitlines()[-1])["val_loss"]
print(f"기록 {row['val_loss']:.4f}, 다시 실행 {again:.4f}, 같은가? {again == row['val_loss']}")
