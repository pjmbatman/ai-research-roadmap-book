# early stopping 규칙을 callback으로 구현하기

trainer는 검증이 끝날 때마다 callback에 `(epoch, value)`를 넘기고, callback은 학습을 멈출지 알려 준다. 클래스 `EarlyStopping`의 `update(epoch, value)`를 완성한다.

| 인자 | 의미 |
|---|---|
| `patience` | 개선이 없는 검증을 몇 번 연속 허용하는가 |
| `min_delta` | 개선으로 인정할 최소 폭(0 이상) |
| `mode` | `"min"`(손실처럼 작을수록 좋음) 또는 `"max"`(정확도처럼 클수록 좋음) |

규칙은 다음과 같다.

1. **개선**: `mode="min"`이면 `value < best - min_delta`, `mode="max"`이면 `value > best + min_delta`. 첫 검증은 항상 개선이다. 개선이면 `best`, `best_epoch`를 갱신하고 기다린 횟수를 0으로 되돌린다.
2. **개선이 아님**: 기다린 횟수를 1 늘린다. 같은 값(동점)도 개선이 아니다. 기다린 횟수가 `patience`에 도달하면 `True`(멈춤)를 반환한다.
3. **발산**: `value`가 NaN이거나 무한대이면 `best`를 바꾸지 않고 즉시 `True`를 반환한다.
4. 그 밖에는 `False`를 반환한다.

예: `patience=2`, `min_delta=0`, `mode="min"`에서 검증 손실이 epoch 1–6에 `1.0, 0.8, 0.85, 0.79, 0.80, 0.81`이면, epoch 6에서 `True`를 반환하고 `best_epoch = 4`, `best = 0.79`다. 같은 손실에 `min_delta=0.02`를 주면 0.79는 0.8보다 0.02 이상 좋아지지 않았으므로 개선이 아니고, epoch 4에서 멈추며 `best_epoch = 2`다.

학습을 멈춘 뒤에는 trainer가 `best_epoch`의 checkpoint를 복원한다(17.6장). 그래서 `best_epoch`가 정확해야 한다.
