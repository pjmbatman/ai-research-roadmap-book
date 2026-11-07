"""17.7 실습: 연구실 템플릿용 작은 trainer.

책임: 학습 루프의 순서, 주기적 검증, callback 호출, 학습 상태의 저장·복원.
책임이 아닌 것: 모델 구조, 데이터, loss와 metric의 정의(과제별 step 함수와 eval 함수가 맡는다).

    trainer = Trainer(model, optimizer, step_fn, eval_fn, callbacks=[...], max_epochs=50)
    trainer.fit(train_loader)

step_fn(model, batch) -> loss 텐서          (forward와 loss만. backward·step은 trainer가 한다)
eval_fn(model) -> dict[str, float]          (평가 모드 전환과 집계는 eval_fn이 책임진다)
"""
from pathlib import Path

import torch


class Callback:
    """필요한 이벤트만 재정의한다. 모든 메서드는 trainer를 첫 인자로 받는다."""

    def on_train_start(self, trainer): ...
    def on_step_end(self, trainer, loss): ...
    def on_eval_end(self, trainer, metrics): ...
    def on_train_end(self, trainer): ...


class Trainer:
    def __init__(self, model, optimizer, step_fn, eval_fn, callbacks=(), max_epochs=10,
                 scheduler=None, grad_clip=None):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.step_fn, self.eval_fn = step_fn, eval_fn
        self.callbacks, self.max_epochs, self.grad_clip = list(callbacks), max_epochs, grad_clip
        self.epoch, self.global_step, self.should_stop = 0, 0, False
        self.last_metrics = {}

    def _emit(self, event, **kw):
        for cb in self.callbacks:
            getattr(cb, event)(self, **kw)

    def fit(self, loader):
        self._emit("on_train_start")
        while self.epoch < self.max_epochs and not self.should_stop:
            self.model.train()
            for batch in loader:
                loss = self.step_fn(self.model, batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip is not None:                       # 기본값은 '하지 않음'
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:                       # step 단위 scheduler
                    self.scheduler.step()
                self.global_step += 1
                self._emit("on_step_end", loss=loss.detach())
            self.epoch += 1
            self.last_metrics = self.eval_fn(self.model)
            self._emit("on_eval_end", metrics=self.last_metrics)
        self._emit("on_train_end")
        return self.last_metrics

    # 재개에 필요한 상태. callback의 상태(예: early stopping의 best)도 함께 저장한다.
    def state_dict(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": self.epoch, "global_step": self.global_step, "rng": torch.get_rng_state(),
                "callbacks": [cb.state_dict() if hasattr(cb, "state_dict") else None for cb in self.callbacks]}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.epoch, self.global_step = state["epoch"], state["global_step"]
        torch.set_rng_state(state["rng"])
        for cb, s in zip(self.callbacks, state["callbacks"]):
            if s is not None:
                cb.load_state_dict(s)


class EarlyStopping(Callback):
    def __init__(self, monitor, patience, min_delta=0.0):
        self.monitor, self.patience, self.min_delta = monitor, patience, min_delta
        self.best, self.wait = float("inf"), 0

    def on_eval_end(self, trainer, metrics):
        if metrics[self.monitor] < self.best - self.min_delta:
            self.best, self.wait = metrics[self.monitor], 0
        else:
            self.wait += 1
            trainer.should_stop = self.wait >= self.patience

    def state_dict(self):
        return {"best": self.best, "wait": self.wait}

    def load_state_dict(self, s):
        self.best, self.wait = s["best"], s["wait"]


class Checkpoint(Callback):
    """매 검증 뒤 last.pt를, monitor가 좋아지면 best.pt를 저장한다. 임시 파일에 쓴 뒤 이름을 바꾼다."""

    def __init__(self, out_dir, monitor, extra=None):
        self.out, self.monitor, self.extra = Path(out_dir), monitor, extra or {}
        self.best = float("inf")

    def _save(self, trainer, name):
        self.out.mkdir(parents=True, exist_ok=True)
        tmp = self.out / f"{name}.tmp"
        torch.save({**trainer.state_dict(), **self.extra, "metrics": trainer.last_metrics}, tmp)
        tmp.replace(self.out / name)

    def on_eval_end(self, trainer, metrics):
        if metrics[self.monitor] < self.best:
            self.best = metrics[self.monitor]
            self._save(trainer, "best.pt")
        self._save(trainer, "last.pt")

    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, s):
        self.best = s["best"]


class PrintLogger(Callback):
    def __init__(self, every_epochs=1):
        self.every = every_epochs

    def on_eval_end(self, trainer, metrics):
        if trainer.epoch % self.every == 0 or trainer.should_stop:
            text = "  ".join(f"{k} {v:.4f}" for k, v in metrics.items())
            print(f"epoch {trainer.epoch:3d}  step {trainer.global_step:5d}  {text}")
