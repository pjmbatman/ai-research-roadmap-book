import numpy as np


def successive_halving(evaluate, n_configs, min_budget=1, eta=3):
    """후보를 적은 예산으로 평가하고 상위 1/eta만 남기며 예산을 eta배로 늘린다.

    evaluate(i, budget) -> 손실(낮을수록 좋음)
    반환: (마지막에 남은 후보 번호, evaluate에 넘긴 budget의 총합)
    """
    raise NotImplementedError("단계마다 후보를 평가하고 상위 max(1, n // eta)개를 남기세요")
