import math


class EarlyStopping:
    """검증 metric을 받아 학습을 멈출지 정하는 callback."""

    def __init__(self, patience, min_delta=0.0, mode="min"):
        if mode not in ("min", "max"):
            raise ValueError("mode는 'min' 또는 'max'")
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = math.inf if mode == "min" else -math.inf
        self.best_epoch = None
        self.wait = 0

    def _improved(self, value):
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def update(self, epoch, value):
        """epoch의 검증 값을 받아, 지금 멈춰야 하면 True를 반환한다."""
        value = float(value)
        if not math.isfinite(value):                 # NaN·inf: 발산
            return True
        if self.best_epoch is None or self._improved(value):
            self.best, self.best_epoch, self.wait = value, epoch, 0
            return False
        self.wait += 1
        return self.wait >= self.patience
