"""6.6 실습: 연구 결론을 바꾸는 버그 세 가지가 숨을 수 있는 작은 파이프라인.

- split_by_group: 같은 시료(group)의 반복 측정이 train과 test에 나뉘지 않게 분할한다.
- mae: batch 단위로 누적해 전체 표본의 평균 절대오차를 계산한다.
- Trainer: 최소제곱 문제를 momentum SGD로 풀고, checkpoint를 저장·복원한다.

환경변수 INJECT_BUG로 일부러 버그를 넣을 수 있다(테스트가 잡는지 확인하는 용도).
  INJECT_BUG=leak        행 단위 무작위 분할(같은 시료가 양쪽에 들어간다)
  INJECT_BUG=metric      batch 평균의 평균(마지막 batch가 작으면 틀린다)
  INJECT_BUG=checkpoint  momentum 상태를 저장하지 않는다(재개한 학습이 달라진다)
"""
import os

import numpy as np

BUG = os.environ.get("INJECT_BUG", "")


def split_by_group(groups, test_frac, seed):
    """groups: (N,) 시료 ID. 반환: (train_idx, test_idx), 각각 정렬된 정수 배열."""
    groups = np.asarray(groups)
    if BUG == "leak":
        idx = np.random.default_rng(seed).permutation(len(groups))
        k = int(round(len(groups) * test_frac))
        return np.sort(idx[k:]), np.sort(idx[:k])
    uniq = np.unique(groups)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_groups = np.random.default_rng(seed).choice(uniq, n_test, replace=False)
    is_test = np.isin(groups, test_groups)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def mae(y_true, y_pred, batch_size):
    """y_true, y_pred: (N,). batch 단위로 누적한 전체 평균 절대오차."""
    total, count, batch_means = 0.0, 0, []
    for s in range(0, len(y_true), batch_size):
        err = np.abs(y_true[s:s + batch_size] - y_pred[s:s + batch_size])   # (≤B,)
        total += err.sum()
        count += err.size
        batch_means.append(err.mean())
    if BUG == "metric":
        return float(np.mean(batch_means))
    return total / count


class Trainer:
    """f(w) = 0.5 * ||A w - b||^2 를 momentum SGD로 최소화한다. A: (N, D), b: (N,)."""

    def __init__(self, A, b, lr=0.01, momentum=0.9):
        self.A, self.b, self.lr, self.momentum = A, b, lr, momentum
        self.w = np.zeros(A.shape[1])      # (D,)
        self.v = np.zeros(A.shape[1])      # (D,) momentum 상태
        self.step = 0

    def loss(self):
        r = self.A @ self.w - self.b
        return 0.5 * float(r @ r)

    def train(self, n_steps):
        for _ in range(n_steps):
            grad = self.A.T @ (self.A @ self.w - self.b)      # (D,)
            self.v = self.momentum * self.v - self.lr * grad
            self.w = self.w + self.v
            self.step += 1

    def save(self, path):
        state = {"w": self.w, "v": self.v, "step": np.array(self.step)}
        if BUG == "checkpoint":
            del state["v"]
        np.savez(path, **state)

    def load(self, path):
        with np.load(path) as ck:
            self.w = ck["w"].copy()
            self.v = ck["v"].copy() if "v" in ck else np.zeros_like(self.w)
            self.step = int(ck["step"])
