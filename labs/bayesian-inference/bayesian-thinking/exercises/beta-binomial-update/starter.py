import numpy as np


def update(a, b, tosses):
    """사전 Beta(a, b)에 관측 tosses(1 = 앞면, 0 = 뒷면)를 반영한 사후 Beta의 파라미터를 반환한다.

    tosses: (n,) 0과 1로 된 배열(빈 배열 가능)
    반환: (a_post, b_post)
    """
    raise NotImplementedError("사후분포의 두 파라미터를 계산하세요")


def predictive_pmf(a, b, m):
    """θ ~ Beta(a, b)일 때 앞으로 m번 던져 앞면이 j번 나올 확률(j = 0..m)을 반환한다.

    반환: (m+1,) 배열, 합이 1
    """
    raise NotImplementedError("Beta-Binomial 확률을 계산하세요")
