RECIPES = {   # (파라미터, gradient, optimizer 상태, 활성값) 원소당 byte
    "fp32-sgd": (4, 4, 0, 4),
    "fp32-adam": (4, 4, 8, 4),
    "bf16-adam": (2, 2, 12, 2),
}


def training_memory(widths, batch, recipe="fp32-adam"):
    """ReLU MLP 학습의 메모리를 byte로 센다.

    widths: 층 폭 [d_0, ..., d_L], batch: batch 크기, recipe: RECIPES의 키
    반환: {"params", "grads", "optimizer", "activations", "total"} → int (byte)
    """
    raise NotImplementedError("파라미터 수를 세고 항목별 byte를 계산하세요")


def max_batch(widths, budget_bytes, recipe="fp32-adam"):
    """total이 budget_bytes 이하가 되는 가장 큰 batch. batch 1도 안 되면 0."""
    raise NotImplementedError("batch와 무관한 부분을 먼저 빼세요")
