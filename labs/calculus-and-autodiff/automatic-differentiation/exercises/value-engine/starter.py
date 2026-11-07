import math


class Value:
    """스칼라 값 하나와 그 gradient를 담는 계산 그래프의 노드."""

    def __init__(self, data, parents=(), local_grads=()):
        self.data, self.grad = float(data), 0.0
        self.parents, self.local_grads = parents, local_grads   # 입력 노드와 국소 미분

    def __add__(self, other):
        """self + other. other가 숫자이면 Value로 감싼다. 국소 미분은 (1, 1)."""
        raise NotImplementedError("덧셈 노드를 만드세요")

    def __mul__(self, other):
        """self * other. 국소 미분은 (other.data, self.data)."""
        raise NotImplementedError("곱셈 노드를 만드세요")

    def tanh(self):
        """tanh(self). 국소 미분은 1 - tanh^2."""
        raise NotImplementedError("tanh 노드를 만드세요")

    def backward(self):
        """self.grad = 1에서 시작해 위상 순서의 역순으로 chain rule을 적용한다."""
        raise NotImplementedError("위상 정렬 후 출력 쪽부터 grad를 누적하세요")
