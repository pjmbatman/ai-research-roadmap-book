import numpy as np


def q_learning_update(Q, s, a, r, s_next, done, alpha, gamma):
    """경험 (s, a, r, s_next) 하나로 Q-learning 갱신을 한 번 하고 새 Q를 반환한다.

    Q: (S, A) 행동 가치 표, s, a, s_next: 정수, r: 보상, done: s_next가 종료 상태인지,
    alpha: 학습률, gamma: 할인율
    반환: 갱신된 Q (S, A). 입력 Q는 바꾸지 않는다.
    """
    Q = np.array(Q, dtype=float, copy=True)
    target = r if done else r + gamma * Q[s_next].max()    # TD 목표
    Q[s, a] += alpha * (target - Q[s, a])
    return Q
