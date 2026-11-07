import copy

import numpy as np


class MomentumSGD:
    """선형 회귀 y ≈ X w를 minibatch momentum SGD와 step decay 학습률로 학습한다.

    X, y는 데이터이므로 checkpoint에 넣지 않는다. 나머지 중 step()의 결과에 영향을 주는 것이 상태다.
    """

    def __init__(self, X, y, lr=0.05, beta=0.9, batch_size=8, seed=0):
        self.X, self.y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)   # (N, D), (N,)
        self.lr, self.beta, self.batch_size = lr, beta, batch_size
        self.w = np.zeros(self.X.shape[1])                    # (D,)
        self.velocity = np.zeros_like(self.w)                 # (D,)
        self.rng = np.random.default_rng(seed)                # minibatch 표집용
        self.step_count = 0

    def step(self):
        idx = self.rng.choice(len(self.X), size=self.batch_size, replace=False)
        Xb, yb = self.X[idx], self.y[idx]
        grad = 2 * Xb.T @ (Xb @ self.w - yb) / self.batch_size
        lr = self.lr * 0.5 ** (self.step_count // 10)         # 10 step마다 학습률 절반
        self.velocity = self.beta * self.velocity + grad
        self.w = self.w - lr * self.velocity
        self.step_count += 1

    def state_dict(self):
        """재개에 필요한 상태를 복사본으로 반환한다(이후 학습이 저장본을 바꾸지 않게)."""
        return {
            "w": self.w.copy(),
            "velocity": self.velocity.copy(),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state):
        """state_dict()가 만든 상태를 복원한다. 받은 dict도 복사해 쓴다."""
        self.w = np.array(state["w"], dtype=float)
        self.velocity = np.array(state["velocity"], dtype=float)
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.step_count = int(state["step_count"])
