# successive halving 구현하기

후보 설정 여러 개를 적은 예산으로 학습해 보고, 성적이 좋은 일부만 남겨 예산을 늘리는 과정을 반복한다. 함수 `successive_halving(evaluate, n_configs, min_budget, eta)`를 완성한다.

- `evaluate(i, budget)`는 후보 `i`(0부터 `n_configs - 1`)를 처음부터 `budget` epoch 학습한 뒤의 검증 손실을 반환한다. 낮을수록 좋다.
- 첫 단계(rung)에서는 모든 후보를 `min_budget`으로 평가한다.
- 각 단계에서 평가가 끝나면 손실이 낮은 순서로 `max(1, 현재 후보 수 // eta)`개만 남긴다. 손실이 같으면 번호가 작은 후보를 먼저 남긴다.
- 다음 단계의 예산은 이전 단계의 `eta`배다.
- 후보가 하나 남은 단계에서 그 후보를 평가하고 끝낸다.

반환값은 `(best, total_budget)`이다. `best`는 마지막에 남은 후보의 번호, `total_budget`은 `evaluate`에 넘긴 `budget`의 총합이다(각 평가는 처음부터 학습한다고 보고 모두 더한다).

예: 후보 27개, `min_budget = 1`, `eta = 3`이면 27개 × 1, 9개 × 3, 3개 × 9, 1개 × 27로 총 108 epoch를 쓴다.

테스트는 일찍 순위가 정해지는 학습 곡선에서는 진짜 최선을 찾는지, 그리고 늦게 좋아지는 후보(late bloomer)가 있을 때는 그 후보를 놓칠 수 있다는 것을 함께 확인한다.
