# 안정한 log-softmax

분류 모델의 cross-entropy는 logits $\mathbf{z} \in \mathbb{R}^K$의 log-softmax

$$
\log \mathrm{softmax}(\mathbf{z})_k = z_k - \log \sum_{j=1}^{K} e^{z_j}
$$

를 쓴다. 정의대로 `np.log(np.exp(z) / np.exp(z).sum())`을 계산하면 logits가 크면 `exp`가 overflow해 `nan`이, 작으면 underflow해 `-inf`가 나온다.

함수 `log_softmax(z, axis=-1)`를 완성한다.

- `z`는 임의 shape의 배열이고, `axis` 방향으로 log-softmax를 계산한다. 반환 shape은 `z`와 같다.
- `z = [1000, 1001, 1002]`나 `z = [-1000, 0]`처럼 큰 값·작은 값이 섞여도 `nan`이나 `inf` 없이 정확한 값을 내야 한다.
- `scipy.special.logsumexp`를 쓰지 않고 `numpy`만으로 구현한다.

힌트: 모든 $z_j$에서 같은 상수 $m$을 빼도 log-softmax는 변하지 않는다. $m = \max_j z_j$로 고르면 지수의 인자가 모두 0 이하가 된다. 축을 유지하려면 `keepdims=True`를 쓴다.
