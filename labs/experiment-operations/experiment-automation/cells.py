"""22.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/experiment-operations/experiment-automation/cells.py
학습은 설정으로 정해지는 가짜 검증 손실로 대신한다.
"""
# %% cell 1 — sweep: 설정 조합과 job manifest
import hashlib
import itertools
import json

import numpy as np

def job_id(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

space = {"lr": [1e-4, 1e-3, 1e-2], "wd": [0.0, 1e-4], "seed": [0, 1, 2]}
grid = [dict(zip(space, values)) for values in itertools.product(*space.values())]
manifest = [{"job_id": job_id(c), **c} for c in grid]
print(len(manifest), "jobs, job_id 중복 없음:", len({j["job_id"] for j in manifest}) == len(manifest))
print(manifest[0])

# 같은 설정은 목록 순서와 무관하게 같은 job_id를 갖는다
space2 = {"lr": [1e-2, 1e-3, 1e-4, 3e-4], "wd": [0.0, 1e-4], "seed": [0, 1, 2]}   # 값 하나 추가, 순서 바꿈
ids2 = {job_id(dict(zip(space2, v))) for v in itertools.product(*space2.values())}
print("기존 18개가 새 sweep에도 같은 ID로 있다:", {j["job_id"] for j in manifest} <= ids2,
      " 새로 돌릴 job:", len(ids2 - {j["job_id"] for j in manifest}))

# %% cell 2 — 같은 예산 9회: grid 대 random (중요한 축이 하나뿐일 때)
def val_loss(log_lr, log_wd, opt):
    """lr이 결과의 대부분을 정하고 wd는 조금만 영향을 준다"""
    return (log_lr - opt) ** 2 + 0.01 * (log_wd + 4.5) ** 2

g_lr, g_wd = np.meshgrid(np.linspace(-5, -1, 3), np.linspace(-6, -3, 3))   # (3, 3) 격자
rng = np.random.default_rng(0)
regret = {"grid": [], "random": []}
for _ in range(2000):                                # 최적 lr이 어디 있는지 모르는 문제 2000개
    opt = rng.uniform(-5, -1)
    regret["grid"].append(val_loss(g_lr, g_wd, opt).min())
    r_lr, r_wd = rng.uniform(-5, -1, 9), rng.uniform(-6, -3, 9)          # (9,), (9,)
    regret["random"].append(val_loss(r_lr, r_wd, opt).min())
print("시도한 서로 다른 lr 값: grid", len(np.unique(g_lr)), " random", 9)
for k, v in regret.items():
    print(f"{k:6s}: 최적과의 손실 차이 평균 {np.mean(v):.4f}, 90% 분위수 {np.quantile(v, 0.9):.4f}")
print("random이 grid보다 나은 문제의 비율:", np.mean(np.array(regret["random"]) < np.array(regret["grid"])))

# %% cell 3 — 조기 중단: 같은 예산(108 epoch)의 random search 대 successive halving
def curve(log_lr, t, lr_sets_speed):
    """최종 손실은 log_lr = -3 근처가 가장 낮다. lr_sets_speed이면 lr이 클수록 빨리 내려간다"""
    final = 0.2 + 0.05 * (log_lr + 3) ** 2
    speed = 0.3 * 10 ** (log_lr + 3) if lr_sets_speed else 0.3
    return final + (1.0 - final) * np.exp(-speed * t)

def successive_halving(cands, lr_sets_speed, min_budget=1, eta=3):
    used, budget = 0, min_budget
    while True:
        losses = curve(cands, budget, lr_sets_speed)   # 살아남은 후보를 budget epoch만큼 학습
        used += budget * len(cands)
        if len(cands) == 1:
            return cands[0], used
        cands = cands[np.argsort(losses)[: max(1, len(cands) // eta)]]
        budget *= eta

for lr_sets_speed in [False, True]:
    rng = np.random.default_rng(1)
    best_rs, best_sh = [], []
    for _ in range(500):
        cands = rng.uniform(-5, -1, 27)                # 후보 27개, 최대 27 epoch
        best_rs.append(curve(cands[:4], 27, lr_sets_speed).min())   # 4개를 끝까지: 4 × 27 = 108
        pick, used = successive_halving(cands, lr_sets_speed)
        best_sh.append(curve(pick, 27, lr_sets_speed))
    name = "큰 lr이 빨리 내려감" if lr_sets_speed else "내려가는 속도 같음 "
    print(f"{name}: 예산 {used} epoch, 최종 손실 평균  random {np.mean(best_rs):.4f}  SH {np.mean(best_sh):.4f}")

# %% cell 4 — 견고한 실행: 재시도, 원자적 저장, 중단 후 재개
import os
import tempfile
from pathlib import Path

OUT = Path(tempfile.mkdtemp())
fail_rng = np.random.default_rng(2)                 # 노드 장애 같은 일시적 실패를 흉내 낸다

class Transient(Exception):
    pass

def run_job(job):
    if fail_rng.random() < 0.3:
        raise Transient("node lost")
    if job["lr"] == 1e-2 and job["wd"] == 0.0:
        raise FloatingPointError("loss is NaN")     # 설정 때문: 다시 돌려도 같다
    return {"val_loss": round(0.2 + 0.05 * (np.log10(job["lr"]) + 3) ** 2, 4)}

def save_atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)                           # 이름 바꾸기는 원자적: 반쯤 쓴 파일이 보이지 않는다

def run_queue(manifest, max_attempts=3, stop_after=None):
    for k, job in enumerate(manifest):
        if k == stop_after:
            return                                  # runner 자체가 죽었다
        path = OUT / f"{job['job_id']}.json"
        if path.exists():
            continue                                # 결과가 있는 job은 건너뛴다
        for attempt in range(1, max_attempts + 1):
            try:
                result = {"status": "finished", **run_job(job)}
                break
            except Transient:
                result = {"status": "failed", "error": "transient"}
            except FloatingPointError as e:
                result = {"status": "failed", "error": str(e)}
                break                               # 결정적 실패는 재시도하지 않는다
        save_atomic(path, {**result, "attempts": attempt})

run_queue(manifest, stop_after=10)
print("중단 시점의 결과 파일:", len(list(OUT.glob("*.json"))))
run_queue(manifest)
results = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
summary = {}
for r in results:
    key = (r["status"], r.get("error", "-"))
    summary[key] = summary.get(key, 0) + 1
print("재개 후:", len(results), "개", dict(sorted(summary.items())))
print("시도 횟수 분포:", {a: sum(r["attempts"] == a for r in results) for a in (1, 2, 3)})
