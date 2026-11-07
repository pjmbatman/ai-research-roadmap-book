import numpy as np


def successive_halving(evaluate, n_configs, min_budget=1, eta=3):
    """후보를 적은 예산으로 평가하고 상위 1/eta만 남기며 예산을 eta배로 늘린다.

    evaluate(i, budget) -> 손실(낮을수록 좋음)
    반환: (마지막에 남은 후보 번호, evaluate에 넘긴 budget의 총합)
    """
    alive = list(range(n_configs))
    budget, total = min_budget, 0
    while True:
        losses = [evaluate(i, budget) for i in alive]
        total += budget * len(alive)
        if len(alive) == 1:
            return alive[0], total
        order = sorted(range(len(alive)), key=lambda k: (losses[k], alive[k]))   # 손실, 같으면 번호 순
        alive = [alive[k] for k in order[: max(1, len(alive) // eta)]]
        budget *= eta
