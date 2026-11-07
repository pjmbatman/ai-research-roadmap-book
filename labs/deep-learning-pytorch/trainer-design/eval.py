"""17.7 실습: checkpoint를 평가한다. trainer.py와 optimizer를 import하지 않는다.

    uv run --project labs --group torch python labs/deep-learning-pytorch/trainer-design/eval.py CKPT [CKPT ...]
"""
import sys

import torch

from task import build_model, make_data, make_eval_fn


def main():
    _, (xv, yv) = make_data()
    eval_fn = make_eval_fn(xv, yv, batch_size=7)                 # 학습 때와 다른 batch 크기
    for path in sys.argv[1:]:
        ckpt = torch.load(path, weights_only=False)                # 직접 만든 파일만 이렇게 연다
        model = build_model(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        metrics = {k: round(v, 4) for k, v in eval_fn(model).items()}
        saved = {k: round(v, 4) for k, v in ckpt["metrics"].items()}
        print(f"{path.split('outputs/')[-1]:8s} epoch {ckpt['epoch']:3d}  다시 계산 {metrics}  저장된 값과 같은가 {metrics == saved}")


if __name__ == "__main__":
    main()
