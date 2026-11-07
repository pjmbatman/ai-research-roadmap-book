"""4.6 실습: 에이전트에게 맡길 작은 ablation 실험.

합성 '재료 물성' 회귀 문제다. 특징은 세 묶음(group)으로 나뉜다.
  composition (3개): 조성. 목표값에 비선형으로 기여한다
  process     (2개): 공정 조건. 목표값에 선형으로 기여한다
  noise       (10개): 목표값과 무관하다
질문: 각 묶음을 뺐을 때 검증 RMSE가 얼마나 나빠지는가?

    python experiment.py                       # 모든 특징
    python experiment.py --drop process        # process 묶음을 빼고
    python experiment.py --seeds 0 1 2 --out outputs/full.json

이 스크립트는 train으로 학습하고 **validation**으로만 평가한다. test 평가는 final_test_eval.py가 한다.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

GROUPS = {"composition": [0, 1, 2], "process": [3, 4], "noise": list(range(5, 15))}
SPLIT_SEED = 0                                         # 분할은 고정한다. 바꾸지 않는다


def make_data(n=1200, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 15))                                         # (N, 15)
    y = (np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2]                         # composition
         + 0.8 * X[:, 3] - 0.5 * X[:, 4]                                 # process
         + 0.3 * rng.normal(size=n))                                     # 측정 잡음
    return X, y


def split(n, seed=SPLIT_SEED):
    """train 60% / val 20% / test 20% 인덱스."""
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return idx[:a], idx[a:b], idx[b:]


def split_fingerprint(test_idx):
    """test 인덱스의 해시. 실행 기록에 남겨 분할이 바뀌지 않았음을 확인한다."""
    return hashlib.sha256(np.sort(test_idx).astype(np.int64).tobytes()).hexdigest()[:12]


def columns(drop):
    dropped = set(GROUPS[drop]) if drop else set()
    return [j for j in range(15) if j not in dropped]


def fit(X, y, seed):
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=1).fit(X, y)


def run(drop=None, seeds=(0, 1, 2)):
    X, y = make_data()
    tr, va, te = split(len(y))
    cols = columns(drop)
    rmses = []
    for s in seeds:
        model = fit(X[tr][:, cols], y[tr], s)
        rmses.append(float(np.sqrt(np.mean((model.predict(X[va][:, cols]) - y[va]) ** 2))))
    return {"drop": drop, "seeds": list(seeds), "val_rmse": rmses,
            "val_rmse_mean": float(np.mean(rmses)), "val_rmse_std": float(np.std(rmses, ddof=1)),
            "n_features": len(cols), "test_fingerprint": split_fingerprint(te)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drop", choices=sorted(GROUPS), default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    res = run(a.drop, tuple(a.seeds))
    print(f"drop={a.drop or '-':12s} val RMSE {res['val_rmse_mean']:.3f} ± {res['val_rmse_std']:.3f} "
          f"(seeds {a.seeds}, test 분할 {res['test_fingerprint']})")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
