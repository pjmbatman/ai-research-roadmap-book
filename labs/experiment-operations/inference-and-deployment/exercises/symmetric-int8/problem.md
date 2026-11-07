# 대칭 int8 양자화

학습된 가중치를 8비트 정수로 저장하면 fp32보다 메모리가 4분의 1로 준다. **대칭 양자화**(symmetric quantization)는 0을 0으로 보내고, 묶음 안의 절댓값 최댓값을 127에 맞춘다.

$$
s = \frac{\max_i \lvert w_i \rvert}{127}, \qquad q_i = \operatorname{clip}\big(\operatorname{round}(w_i / s),\ -127,\ 127\big), \qquad \hat{w}_i = s\,q_i.
$$

여기서 $s$는 scale, $q_i$는 int8 정수, $\hat{w}_i$는 복원한 값이다. -128은 쓰지 않아 양수와 음수의 범위가 대칭이 된다.

두 함수를 완성한다.

1. `quantize(w, axis=None)` — `(q, scale)`를 반환한다.
   - `axis=None`: 텐서 전체에 scale 하나(per-tensor). `scale`은 `float`.
   - `axis=0`: 2차원 `w`의 **행마다** scale 하나(per-channel, 행 = 출력 채널). `scale`의 shape은 `(행 수, 1)`이라 `q * scale`이 broadcasting으로 바로 계산된다.
   - `q`는 `w`와 shape이 같은 `np.int8` 배열이다.
   - 묶음이 모두 0이면 최댓값이 0이라 0으로 나누게 된다. 이때는 scale을 `1.0`으로 둔다(그러면 `q`도 모두 0이다).
2. `dequantize(q, scale)` — $\hat{w} = s\,q$를 실수 배열로 반환한다.

테스트가 확인하는 성질은 다음과 같다. 반올림 오차는 원소마다 $s/2$ 이하이고, 절댓값이 가장 큰 원소는 정확히 ±127이 되며, 행마다 크기가 크게 다른 가중치에서는 행별 scale의 오차가 텐서 하나의 scale보다 작다.

힌트: `np.abs(w).max(axis=1, keepdims=True)`는 shape `(행 수, 1)`인 행별 최댓값을 준다. 정수로 바꾸는 `astype(np.int8)`은 범위를 넘는 값을 감아 버리므로(5.5장), 반드시 먼저 `np.clip`으로 자른다.
