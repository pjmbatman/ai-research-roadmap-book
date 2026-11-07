"""17.6 실습: checkpoint 하나만 받아 검증 MAE를 다시 계산하는 독립 평가 스크립트.

    uv run --project labs --group torch python labs/deep-learning-pytorch/evaluation-and-checkpoints/eval.py CKPT [CKPT ...]

학습 루프·optimizer를 import하지 않는다. 모델 구조는 checkpoint의 config로, 전처리는 contract로 다시 만든다.
"""
import hashlib
import sys

import torch

from common import build_model, evaluate, make_data


def main():
    _, (Xv, yv) = make_data()
    for path in sys.argv[1:]:
        ckpt = torch.load(path, weights_only=False)
        model = build_model(ckpt["config"])
        model.load_state_dict(ckpt["model"])                             # strict=True: 이름·shape이 모두 맞아야 한다
        digest = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()[:12]
        mae = evaluate(model, Xv, yv, ckpt["contract"])
        print(f"{path.split('outputs/')[-1]:20s} step {ckpt['step']:4d}  검증 MAE {mae:.4f}  파라미터 해시 {digest}")


if __name__ == "__main__":
    main()
