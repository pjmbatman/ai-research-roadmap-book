def run_ops(ops, w0, lr, grad_fn):
    """zero_grad / backward / step 연산열을 스칼라 파라미터 하나에 대해 실행한다.

    ops: 문자열 list, w0: 초기 파라미터, lr: 학습률, grad_fn: w -> dL/dw
    반환: (w, grad)  grad는 None일 수 있다
    """
    raise NotImplementedError("w와 grad 버퍼를 두고 연산을 차례로 실행하세요")
