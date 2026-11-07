"""22.7 실습: 학습한 모델을 내보내고, 원본과 출력 일치·속도를 비교한다.

이 실습은 PyTorch에 포함된 torch.export를 쓴다. ONNX로 내보내려면 onnx(와 실행용 onnxruntime) 패키지가
따로 필요하다(본문의 집필 시점 예시 참고). 비교 절차는 같다.

실행: uv run --project labs --group torch python labs/experiment-operations/inference-and-deployment/export_check.py
"""
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).parent / "outputs"
torch.set_num_threads(1)


def train_model():
    torch.manual_seed(0)
    X = torch.rand(2000, 6) * 2 - 1                                  # (N, 6)
    y = torch.sin(3 * X[:, :1]) + X[:, 1:2] * X[:, 2:3]               # (N, 1)
    model = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 128),
                          nn.GELU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        idx = torch.randint(0, len(X), (128,))
        loss = nn.functional.mse_loss(model(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def best_ms(fn, x, repeat=20):
    with torch.inference_mode():
        for _ in range(3):                                            # warm-up
            fn(x)
        ts = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn(x)
            ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


def main():
    model = train_model()
    example = torch.randn(4, 6)
    ep = torch.export.export(model, (example,), dynamic_shapes=({0: torch.export.Dim("batch")},))
    OUT.mkdir(exist_ok=True)
    path = OUT / "model.pt2"
    torch.export.save(ep, path)
    deployed = torch.export.load(path).module()
    print(f"내보낸 파일 {path.name}: {path.stat().st_size / 1024:.0f} KiB")

    # 1) 수치 일치: 학습 분포 안과 밖의 입력 모두
    torch.manual_seed(1)
    for name, x in [("분포 안 [-1, 1]", torch.rand(10_000, 6) * 2 - 1), ("분포 밖 [-5, 5]", torch.rand(10_000, 6) * 10 - 5)]:
        with torch.inference_mode():
            ref, out = model(x), deployed(x)
        diff = (ref - out).abs().max().item()
        print(f"{name}: 최대 절대 차이 {diff:.2e}, allclose(rtol=1e-5, atol=1e-6) {torch.allclose(ref, out, rtol=1e-5, atol=1e-6)}")

    # 2) 속도: batch 크기별 지연과 처리량 (시간은 기계마다, 실행마다 다르다)
    print("batch   원본(ms)  내보낸 모델(ms)  처리량(표본/s, 내보낸 모델)")
    for B in [1, 32, 1024]:
        x = torch.randn(B, 6)
        t_eager, t_exp = best_ms(model, x), best_ms(deployed, x)
        print(f"{B:5d} {t_eager:10.3f} {t_exp:16.3f} {B / t_exp * 1000:20.0f}")


if __name__ == "__main__":
    main()
