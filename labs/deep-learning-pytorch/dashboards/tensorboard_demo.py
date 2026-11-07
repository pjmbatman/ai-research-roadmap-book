"""17.8 실습: TensorBoard 로그를 쓰고, 기록된 원본을 다시 읽어 대시보드가 보여 주는 것과 비교한다.

tensorboard 패키지는 실습 공통 환경에 들어 있지 않다. uv의 --with로 이 실행에만 더한다(집필 시점 예시).

    uv run --project labs --group torch --with tensorboard python labs/deep-learning-pytorch/dashboards/tensorboard_demo.py
    uv run --project labs --group torch --with tensorboard tensorboard --logdir labs/deep-learning-pytorch/dashboards/outputs/tb

20000 step의 손실을 기록하고, step 12345에 한 번 spike를 넣는다. TensorBoard의 이벤트 읽기 도구는
기본 설정에서 scalar를 태그마다 일정 개수까지만 무작위로 남기므로(reservoir sampling), spike가 빠질 수 있다.
"""
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

OUT = Path(__file__).parent / "outputs" / "tb" / "run1"

torch.manual_seed(0)
model = nn.Linear(4, 1)
writer = SummaryWriter(log_dir=str(OUT))
for step in range(20000):
    loss = 1.0 / (1 + step / 100) + (5.0 if step == 12345 else 0.0)      # 손실 곡선을 흉내 낸 값
    writer.add_scalar("train/loss", loss, step)
    writer.add_scalar("train/lr", 1e-3 * min(1.0, step / 1000), step)
    if step % 5000 == 0:
        writer.add_histogram("weights/linear", model.weight.detach(), step)
writer.add_text("notes", "spike를 step 12345에 넣었다", 0)
writer.add_hparams({"lr": 1e-3, "batch_size": 32}, {"hparam/final_loss": loss})
writer.close()

print("기본 한도:", {k: v for k, v in event_accumulator.DEFAULT_SIZE_GUIDANCE.items() if k in ("scalars", "histograms")})
for guidance, label in [(None, "기본 설정"), ({"scalars": 0}, "전부 읽기")]:
    acc = event_accumulator.EventAccumulator(str(OUT), size_guidance=guidance)
    acc.Reload()
    events = acc.Scalars("train/loss")
    print(f"{label}: 읽은 점 {len(events):5d}개  최댓값 {max(e.value for e in events):.3f}  "
          f"step 12345 포함 {any(e.step == 12345 for e in events)}")
print("태그:", sorted(acc.Tags()["scalars"]), sorted(acc.Tags()["histograms"]))
