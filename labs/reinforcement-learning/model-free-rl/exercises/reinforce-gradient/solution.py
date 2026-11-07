import numpy as np


def reinforce_grad(theta, actions, returns, baseline=0.0):
    """softmax 정책 pi = softmax(theta)에 대한 REINFORCE 기울기 추정량의 표본 평균.

    theta: (A,) 로짓, actions: (N,) 정수 행동, returns: (N,) 각 행동 뒤의 이득 G,
    baseline: 이득에서 뺄 상수 b
    반환: (A,) 배열, mean_i (G_i - b) * grad log pi(a_i)
    """
    theta = np.asarray(theta, dtype=float)
    actions = np.asarray(actions)
    returns = np.asarray(returns, dtype=float)
    z = np.exp(theta - theta.max())          # 최댓값을 빼 overflow 방지
    pi = z / z.sum()                         # (A,)
    score = np.eye(theta.size)[actions] - pi # grad log pi(a) = onehot(a) - pi, (N, A)
    return ((returns - baseline)[:, None] * score).mean(axis=0)
