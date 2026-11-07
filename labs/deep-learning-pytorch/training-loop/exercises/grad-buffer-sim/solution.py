def run_ops(ops, w0, lr, grad_fn):
    """zero_grad / backward / step 연산열을 스칼라 파라미터 하나에 대해 실행한다.

    ops: 문자열 list, w0: 초기 파라미터, lr: 학습률, grad_fn: w -> dL/dw
    반환: (w, grad)  grad는 None일 수 있다
    """
    w, grad = float(w0), None
    for op in ops:
        if op == "zero_grad":
            grad = None
        elif op == "backward":
            g = grad_fn(w)                          # 현재 w에서의 gradient
            grad = g if grad is None else grad + g  # 버퍼에 더한다
        elif op == "step":
            if grad is not None:
                w = w - lr * grad
        else:
            raise ValueError(f"알 수 없는 연산: {op!r}")
    return w, grad
