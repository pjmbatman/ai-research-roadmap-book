import numpy as np


def train_linear(X, y, micro_batch, accum, lr, epochs, seed):
    """gradient accumulation이 있는 선형회귀 SGD.

    X: (N, D), y: (N,)
    반환: (w (D,), b float, n_steps int)
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    N, D = X.shape
    w, b, n_steps = np.zeros(D), 0.0, 0
    for epoch in range(epochs):
        order = np.random.default_rng([seed, epoch]).permutation(N)
        micro = [order[i:i + micro_batch] for i in range(0, N, micro_batch)]
        gw_sum, gb_sum, count = np.zeros(D), 0.0, 0                 # zero_grad
        for k, idx in enumerate(micro, 1):
            r = X[idx] @ w + b - y[idx]                             # 잔차 (n,)
            gw_sum += 2 * X[idx].T @ r                              # 표본별 gradient의 합
            gb_sum += 2 * r.sum()
            count += len(idx)
            if k % accum == 0 or k == len(micro):                   # accum개마다, epoch 끝에서
                w = w - lr * gw_sum / count                         # 평균 손실의 gradient로 갱신
                b = b - lr * gb_sum / count
                n_steps += 1
                gw_sum, gb_sum, count = np.zeros(D), 0.0, 0         # zero_grad
    return w, b, n_steps
