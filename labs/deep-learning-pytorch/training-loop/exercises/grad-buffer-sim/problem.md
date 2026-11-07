# `.grad` 버퍼 시뮬레이터로 학습 루프의 순서 버그 찾기

PyTorch 학습 루프의 세 호출 `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`은 모두 파라미터마다 붙어 있는 **`.grad` 버퍼**를 읽거나 쓴다. 순서를 바꾸거나 하나를 빠뜨려도 오류는 나지 않는다. 버퍼의 규칙만 알면 결과를 예측할 수 있다. 파라미터가 스칼라 하나인 경우의 시뮬레이터를 만든다.

`run_ops(ops, w0, lr, grad_fn)`을 완성한다.

- 상태: 파라미터 `w`(처음 `w0`)와 버퍼 `grad`(처음 `None`).
- `ops`의 문자열을 차례로 실행한다.
  - `"zero_grad"`: `grad = None` (집필 시점 PyTorch 기본값 `set_to_none=True`와 같다)
  - `"backward"`: `g = grad_fn(w)`를 계산해 버퍼에 **더한다**. 버퍼가 `None`이면 `g`가 된다.
  - `"step"`: 버퍼가 `None`이면 아무것도 하지 않는다. 아니면 `w = w - lr * grad` (SGD). 버퍼는 그대로 둔다.
  - 그 밖의 문자열이면 `ValueError`.
- 반환: `(w, grad)` — 모든 연산 뒤의 파라미터와 버퍼.

`backward`는 **현재** `w`에서 gradient를 계산한다는 점에 주의한다. 테스트는 다음 경우를 확인한다.

| 한 반복의 순서 | 결과 |
|---|---|
| `zero_grad, backward, step` | 올바른 SGD |
| `backward, step` (zero_grad 누락) | 과거 gradient가 계속 쌓인다 |
| `backward, zero_grad, step` | 버퍼가 비어 파라미터가 전혀 바뀌지 않는다 |
| `zero_grad, backward, backward, step` | gradient accumulation: 두 gradient의 합으로 한 번 갱신 |
