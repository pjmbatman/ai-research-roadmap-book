# logit에서 perplexity 계산하기

언어모델이 길이 $T$인 열의 각 위치에서 낸 logit $\mathbf{Z} \in \mathbb{R}^{T \times V}$와 정답 토큰 $y_1, \dots, y_T$가 주어졌을 때

$$
\mathrm{PPL} = \exp\Big(-\frac{1}{T}\sum_{t=1}^{T} \log \operatorname{softmax}(\mathbf{z}_t)_{y_t}\Big)
$$

를 계산하는 함수 `perplexity(logits, targets)`를 완성한다.

| 인자 | shape | 설명 |
|---|---|---|
| `logits` | `(T, V)` | 위치마다 사전 크기 $V$의 logit |
| `targets` | `(T,)` | 정답 토큰 번호(0부터 $V-1$) |

반환값은 float 하나다.

조건:

- softmax를 그대로 계산하면 logit이 1000처럼 클 때 `exp`가 넘친다. 로그 확률을 **log-sum-exp**로 안정적으로 계산한다:
  $\log \operatorname{softmax}(\mathbf{z})_i = z_i - m - \log\sum_j \exp(z_j - m)$, $m = \max_j z_j$.
- 모든 logit에 같은 상수를 더해도 결과가 같아야 한다.

테스트는 균등한 logit(답은 $V$), 거의 확신하는 logit(답은 1에 가깝다), 아주 큰 logit, 손으로 계산한 값을 확인한다.

힌트: 정답 위치의 로그 확률만 뽑을 때는 `logp[np.arange(T), targets]`를 쓴다.
