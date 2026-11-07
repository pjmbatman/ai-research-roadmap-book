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

    def update(self, epoch, value):
        """epoch의 검증 값을 받아, 지금 멈춰야 하면 True를 반환한다."""
        raise NotImplementedError("개선 판정, 기다린 횟수, 발산 처리를 구현하세요")
