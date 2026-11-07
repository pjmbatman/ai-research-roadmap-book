"""8.4 실습: 스칼라 reverse mode 자동미분 엔진.

본문 cell 2의 엔진에 연산을 더하고 작은 MLP를 만들 수 있게 한 것이다. NumPy도 쓰지 않는다.
각 연산은 순전파 때 출력값과 함께 '입력에 대한 국소 미분'을 기록하고,
backward()는 위상 정렬한 그래프를 출력에서 입력 쪽으로 훑으며 chain rule을 적용한다.
"""
import math
import random


class Value:
    """스칼라 값 하나와 그 gradient를 담는 계산 그래프의 노드."""

    __slots__ = ("data", "grad", "parents", "local_grads")

    def __init__(self, data, parents=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents          # 이 노드를 만든 입력 노드들
        self.local_grads = local_grads  # 각 입력에 대한 국소 미분 d(self)/d(parent)

    # ------------------------------------------------------------ 기본 연산
    @staticmethod
    def lift(x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = Value.lift(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = Value.lift(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "지수는 상수만 지원한다"
        return Value(self.data**k, (self,), (k * self.data ** (k - 1),))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1.0 - t * t,))

    def relu(self):
        return Value(max(0.0, self.data), (self,), (1.0 if self.data > 0 else 0.0,))

    # ------------------------------------------------------------ 파생 연산
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Value.lift(other))

    def __truediv__(self, other):
        return self * Value.lift(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value.lift(other) - self

    def __rtruediv__(self, other):
        return Value.lift(other) / self

    # ------------------------------------------------------------ 역전파
    def backward(self):
        order, seen = [], set()
        stack = [(self, False)]
        while stack:                     # 반복문으로 쓴 위상 정렬(깊은 그래프에서 재귀 한도를 피한다)
            v, done = stack.pop()
            if done:
                order.append(v)
            elif id(v) not in seen:
                seen.add(id(v))
                stack.append((v, True))
                stack.extend((p, False) for p in v.parents)
        self.grad = 1.0
        for v in reversed(order):
            for p, local in zip(v.parents, v.local_grads):
                p.grad += local * v.grad

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"


class MLP:
    """tanh 은닉층을 가진 완전연결 신경망. 가중치 shape은 (출력, 입력) 리스트."""

    def __init__(self, sizes, seed=0):
        rnd = random.Random(seed)
        self.layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            W = [[Value(rnd.gauss(0, 1) / math.sqrt(n_in)) for _ in range(n_in)] for _ in range(n_out)]
            b = [Value(0.0) for _ in range(n_out)]
            self.layers.append((W, b))

    def __call__(self, x):
        h = [Value.lift(v) for v in x]
        for i, (W, b) in enumerate(self.layers):
            a = [sum((w * hj for w, hj in zip(row, h)), bi) for row, bi in zip(W, b)]
            h = a if i == len(self.layers) - 1 else [ai.tanh() for ai in a]
        return h

    def parameters(self):
        return [p for W, b in self.layers for row in W for p in row] + [p for _, b in self.layers for p in b]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
