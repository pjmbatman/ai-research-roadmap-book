"""17.7 실습: 직접 쓴 학습 루프와 trainer.py의 결과 일치, callback 구성, 중단·재개를 확인한다.

    uv run --project labs --group torch python labs/deep-learning-pytorch/trainer-design/run.py
"""
from pathlib import Path

import torch

from task import CONFIG, batches, build_model, make_data, make_eval_fn, regression_step
from trainer import Checkpoint, EarlyStopping, PrintLogger, Trainer

OUT = Path(__file__).parent / "outputs"
(x, y), (xv, yv) = make_data()
loader, eval_fn = batches(x, y, CONFIG["batch_size"]), make_eval_fn(xv, yv)


def setup():
    torch.manual_seed(CONFIG["seed"])
    model = build_model(CONFIG)
    return model, torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])


def plain_loop(epochs):
    model, opt = setup()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def make_trainer(max_epochs, callbacks=()):
    model, opt = setup()
    return Trainer(model, opt, regression_step, eval_fn, callbacks, max_epochs=max_epochs)


def same(m1, m2):
    return all(torch.equal(p, q) for p, q in zip(m1.parameters(), m2.parameters()))


print("[1] 직접 쓴 루프와 Trainer, 20 epoch")
t = make_trainer(20)
t.fit(loader)
print("    파라미터 일치:", same(plain_loop(20), t.model), "  metric:", {k: round(v, 4) for k, v in t.last_metrics.items()})

print("[2] EarlyStopping + Checkpoint + PrintLogger")
t = make_trainer(300, [EarlyStopping("val_rmse", patience=10, min_delta=1e-3),
                       Checkpoint(OUT, "val_rmse", extra={"config": CONFIG}), PrintLogger(every_epochs=10)])
t.fit(loader)
for min_delta in (1e-4, 0.0):
    t = make_trainer(300, [EarlyStopping("val_rmse", patience=10, min_delta=min_delta)])
    m = t.fit(loader)
    print(f"    min_delta {min_delta:g}: 멈춘 epoch {t.epoch}  val_rmse {m['val_rmse']:.4f}")

print("[3] 15 epoch에서 저장하고 새 Trainer로 재개, 끊김 없는 40 epoch와 비교")
straight = make_trainer(40, [EarlyStopping("val_rmse", patience=10)])
straight.fit(loader)
first = make_trainer(15, [EarlyStopping("val_rmse", patience=10)])
first.fit(loader)
state = first.state_dict()
resumed = make_trainer(40, [EarlyStopping("val_rmse", patience=10)])
resumed.load_state_dict(state)
resumed.fit(loader)
print("    파라미터 일치:", same(straight.model, resumed.model),
      "  early stopping 상태 일치:", straight.callbacks[0].state_dict() == resumed.callbacks[0].state_dict())
