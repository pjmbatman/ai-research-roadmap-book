import math


class Value:
    """스칼라 값 하나와 그 gradient를 담는 계산 그래프의 노드."""

    def __init__(self, data, parents=(), local_grads=()):
        self.data, self.grad = float(data), 0.0
        self.parents, self.local_grads = parents, local_grads   # 입력 노드와 국소 미분

    def __add__(self, other):
        """self + other. other가 숫자이면 Value로 감싼다. 국소 미분은 (1, 1)."""
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        """self * other. 국소 미분은 (other.data, self.data)."""
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def tanh(self):
        """tanh(self). 국소 미분은 1 - tanh^2."""
        t = math.tanh(self.data)
        return Value(t, (self,), (1.0 - t * t,))

    def backward(self):
        """self.grad = 1에서 시작해 위상 순서의 역순으로 chain rule을 적용한다."""
        order, seen = [], set()

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v.parents:
                    visit(p)
                order.append(v)

        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            for p, local in zip(v.parents, v.local_grads):
                p.grad += local * v.grad
