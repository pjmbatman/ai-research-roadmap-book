# mask로 손실 집계하기

길이가 다른 시퀀스를 padding해 만든 batch에서, token별 손실 `loss`(shape `(B, T)`)와 실제 token 위치를 표시한 `mask`(shape `(B, T)`, `bool`)가 주어진다. padding 위치의 `loss` 값은 **아무 값이나 들어 있을 수 있다**(0이라는 보장이 없다. `nan`일 수도 있다).

두 함수를 완성한다.

1. `masked_token_mean(loss, mask)`: 실제 token 전체에 대한 평균. 모든 token이 같은 비중을 갖는다.
   $$
   \frac{\sum_{b,t} m_{bt}\,\ell_{bt}}{\sum_{b,t} m_{bt}}
   $$
2. `masked_sequence_mean(loss, mask)`: 시퀀스마다 실제 token의 평균을 낸 뒤, 그 값들을 다시 평균한다. 모든 시퀀스가 같은 비중을 갖는다. 실제 token이 하나도 없는 시퀀스는 평균에서 뺀다.

테스트는 다음을 확인한다.

- padding 위치의 값(큰 수, `nan`)이 결과에 영향을 주지 않는다.
- padding 길이를 늘려도 결과가 같다.
- 길이가 다른 시퀀스에서 두 함수의 값이 본문처럼 서로 다를 수 있다.
- **분할 일관성**: batch를 두 micro-batch로 나누어 계산한 결과를, token 합과 token 수를 따로 모아 합치면 전체 batch의 `masked_token_mean`과 같다. 테스트는 이를 위해 `masked_token_sum_count(loss, mask)`도 부른다. 이 함수는 `(token 손실의 합, token 수)`를 반환한다.

힌트: `np.where(mask, loss, 0.0)`은 mask가 거짓인 곳의 `nan`도 0으로 바꾼다. `loss * mask`는 `nan * 0 = nan`이므로 쓰면 안 된다.
