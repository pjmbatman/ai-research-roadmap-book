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
    p_b, g_b, s_b, a_b = RECIPES[recipe]
    n = sum(i * o + o for i, o in zip(widths[:-1], widths[1:]))     # 가중치 + bias
    out = {
        "params": n * p_b,
        "grads": n * g_b,
        "optimizer": n * s_b,
        "activations": batch * sum(widths[:-1]) * a_b,             # 각 Linear의 입력
    }
    out["total"] = sum(out.values())
    return out


def max_batch(widths, budget_bytes, recipe="fp32-adam"):
    """total이 budget_bytes 이하가 되는 가장 큰 batch. batch 1도 안 되면 0."""
    static = training_memory(widths, 0, recipe)["total"]           # batch와 무관한 부분
    per_sample = training_memory(widths, 1, recipe)["activations"]
    if budget_bytes < static + per_sample:
        return 0
    return int((budget_bytes - static) // per_sample)
