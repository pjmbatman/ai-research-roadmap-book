import numpy as np


def leapfrog(theta, r, grad_log_p, eps, L):
    """leapfrog L걸음을 진행한 (theta, r). 입력 배열은 바꾸지 않는다."""
    raise NotImplementedError("운동량 반 걸음 → 위치 한 걸음 → 운동량 반 걸음을 L번 반복하세요")


def hamiltonian(theta, r, log_p):
    """H = -log p(theta) + r·r / 2"""
    raise NotImplementedError
