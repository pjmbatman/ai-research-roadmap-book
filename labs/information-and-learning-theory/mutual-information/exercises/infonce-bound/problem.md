# 점수 행렬에서 InfoNCE 계산하기

짝지어진 표본 $K$쌍 $(x_i, y_i)$에 대해 critic이 매긴 점수 행렬 `scores`(shape `(K, K)`)가 주어진다. `scores[i, j]` $= f(x_i, y_j)$이고, 대각선이 진짜 짝의 점수다.

`infonce(scores)`를 완성해 다음 값을 반환한다(nat).

$$
I_{\text{NCE}} = \frac{1}{K}\sum_{i=1}^{K}\left[f(x_i, y_i) - \log\Big(\frac{1}{K}\sum_{j=1}^{K} e^{f(x_i, y_j)}\Big)\right].
$$

점수가 수백 단위로 커질 수 있으므로 `np.exp`를 그대로 쓰면 overflow가 난다. 행마다 최댓값을 빼고 계산하는 log-sum-exp 기법을 쓴다.

$$
\log\sum_j e^{s_j} = m + \log\sum_j e^{s_j - m}, \qquad m = \max_j s_j.
$$

테스트는 다음을 확인한다.

- 모든 점수가 같으면 0이다(critic이 진짜 짝을 구별하지 못함).
- 어떤 점수 행렬에서도 $\log K$를 넘지 않고, 대각선이 압도적으로 크면 $\log K$에 다가간다.
- 점수가 1000 단위여도 `inf`나 `nan` 없이 계산된다.
- 점수 행렬의 각 행에 같은 상수를 더해도 값이 변하지 않는다.
- 행마다 진짜 짝을 정답으로 하는 softmax cross-entropy 평균을 $L$이라 하면 $I_{\text{NCE}} = \log K - L$이다. contrastive 손실 $L$을 줄이는 것이 MI 하한을 올리는 것과 같은 이유다.
