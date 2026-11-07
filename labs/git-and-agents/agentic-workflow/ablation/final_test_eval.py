"""4.6 실습: 최종 test 평가. **사람이** 실험 계획이 확정된 뒤 한 번만 실행한다.

    python final_test_eval.py --drop process

실행할 때마다 test_access.log에 시각·commit·설정을 한 줄 남긴다. 리뷰할 때 이 기록으로
test set이 몇 번, 언제, 어떤 코드로 평가되었는지 확인한다.
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from experiment import GROUPS, columns, fit, make_data, split, split_fingerprint  # noqa: E402

LOG = HERE / "test_access.log"


def current_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "."], cwd=HERE, capture_output=True, text=True)
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def evaluate(drop=None, seed=0, log_path=LOG):
    X, y = make_data()
    tr, va, te = split(len(y))
    trva = np.concatenate([tr, va])                    # 계획이 확정되었으므로 train+val로 다시 학습
    cols = columns(drop)
    model = fit(X[trva][:, cols], y[trva], seed)
    rmse = float(np.sqrt(np.mean((model.predict(X[te][:, cols]) - y[te]) ** 2)))
    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "commit": current_commit(),
             "drop": drop, "seed": seed, "test_fingerprint": split_fingerprint(te), "test_rmse": rmse}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--drop", choices=sorted(GROUPS), default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    e = evaluate(a.drop, a.seed)
    n = sum(1 for _ in open(LOG, encoding="utf-8"))
    print(f"test RMSE {e['test_rmse']:.3f} (drop={a.drop or '-'}, commit {e['commit']}) — test 접근 기록 {n}건")
