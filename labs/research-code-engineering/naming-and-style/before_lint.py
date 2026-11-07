"""6.2 실습용: 일부러 규칙을 어긴 코드. 고치지 말고 도구로 검사해 본다.

집필 시점 예시: ruff check --select E,F,I,B,N before_lint.py
"""
import os
import numpy as np


def LoadCycles(path, cache={}):
    if path in cache:
        return cache[path]
    tmp = np.loadtxt(path)
    cache[path] = tmp
    return tmp


def energy(V, I, t):
    l = V*I*t
    return l.sum()/3600
