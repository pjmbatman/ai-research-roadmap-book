import numpy as np


def train_step(params, X, y, lr):
    """tanh 은닉층 하나의 MLP를 MSE로 한 step 학습한다.

    params: {"W1": (D, H), "b1": (H,), "W2": (H, 1), "b2": (1,)}
    X: (N, D), y: (N, 1), lr: 학습률
    반환: (new_params, loss) — new_params는 새 dict, loss는 갱신 전 파라미터에서의 MSE
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    N = X.shape[0]

    H = np.tanh(X @ W1 + b1)                 # forward  (N, H)
    y_hat = H @ W2 + b2                      #          (N, 1)
    loss = float(np.mean((y_hat - y) ** 2))  # loss

    d_yhat = 2.0 * (y_hat - y) / N           # backward (N, 1)
    dW2 = H.T @ d_yhat                       # (H, 1)
    db2 = d_yhat.sum(axis=0)                 # (1,)
    dA = (d_yhat @ W2.T) * (1.0 - H ** 2)    # (N, H)  tanh'(a) = 1 - tanh(a)^2
    dW1 = X.T @ dA                           # (D, H)
    db1 = dA.sum(axis=0)                     # (H,)

    new_params = {                           # update: 새 배열을 만든다(제자리 수정 없음)
        "W1": W1 - lr * dW1,
        "b1": b1 - lr * db1,
        "W2": W2 - lr * dW2,
        "b2": b2 - lr * db2,
    }
    return new_params, loss
