# broadcasting의 gradient 되돌리기

`y = x + b`에서 `b`의 shape이 `(1, 3)`이고 `x`가 `(4, 3)`이면, `b`는 4개 행에 복제되어 더해진다. 같은 `b`가 여러 번 쓰였으므로 chain rule에 따라 각 사용처의 gradient를 **더해야** 한다(8.4장의 `+=`). autograd는 이 일을 자동으로 하지만, 직접 역전파를 쓰거나 custom 연산을 만들 때는 손으로 해야 한다.

함수 `unbroadcast(grad, shape)`를 완성한다.

- `grad`: broadcast된 결과(출력) 쪽 gradient. shape은 출력의 shape이다.
- `shape`: 원래 입력의 shape(tuple).
- 반환: `shape` 모양의 gradient. 출력에서 broadcast로 늘어난 축을 모두 합해 줄인다.

NumPy broadcasting 규칙(5.3장)을 떠올린다.

1. 입력의 차원 수가 적으면 **앞쪽**에 길이 1인 축이 붙은 것으로 본다. 이렇게 앞에 붙은 축은 합한 뒤 없앤다.
2. 입력에서 길이가 1이었던 축이 출력에서 늘어났다면, 그 축을 합하되 길이 1로 남긴다(`keepdims`).

| `grad.shape` | `shape` | 반환 shape |
|---|---|---|
| `(4, 3)` | `(1, 3)` | `(1, 3)` |
| `(4, 3)` | `(3,)` | `(3,)` |
| `(2, 4, 3)` | `(4, 1)` | `(4, 1)` |
| `(5,)` | `()` (스칼라) | `()` |

테스트는 위 표의 경우들과 무작위 shape에서 수치미분과 비교한다.
