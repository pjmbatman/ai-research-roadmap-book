# softmax cross-entropy와 그 gradient

batch의 logit 행렬 $\mathbf{Z} \in \mathbb{R}^{N \times K}$와 정답 클래스 `y`(길이 $N$의 정수 배열)가 주어진다. $n$번째 샘플의 확률은 $\mathbf{p}_n = \operatorname{softmax}(\mathbf{z}_n)$이고, 손실은 batch 평균 cross-entropy

$$
L = -\frac{1}{N}\sum_{n=1}^{N} \log p_{n, y_n}
$$

이다. 함수 `softmax_ce(Z, y)`를 완성해 `(L, dZ)`를 반환한다. `dZ`는 $\partial L / \partial \mathbf{Z}$로 shape이 `Z`와 같은 `(N, K)`다.

요구 사항:

- 본문에서 유도한 $\partial L / \partial \mathbf{z}_n = (\mathbf{p}_n - \mathbf{y}_n)/N$을 쓴다. 여기서 $\mathbf{y}_n$은 one-hot 벡터다.
- logit이 1000처럼 커도 `nan`이나 `inf`가 나오지 않아야 한다. 행마다 최댓값을 빼고 계산하면 softmax 값은 그대로이고 overflow를 피할 수 있다.

테스트는 유한차분과 비교하고, 큰 logit에서의 안정성, 모든 logit이 같을 때 손실이 $\log K$인지, `dZ`의 각 행의 합이 0인지 확인한다.

힌트: $\log p_{n,k} = z_{n,k} - \log\sum_j e^{z_{n,j}}$로 계산하면 $\log(0)$을 피할 수 있다.
