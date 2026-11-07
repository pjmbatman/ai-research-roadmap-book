# NT-Xent(대칭 InfoNCE) 손실

batch의 표본 $N$개에 증강을 두 번 적용해 얻은 embedding `z1`, `z2`(각각 shape `(N, D)`)가 주어진다. `z1[i]`와 `z2[i]`가 같은 표본의 두 view(positive 쌍)다.

함수 `nt_xent(z1, z2, tau)`를 완성해 SimCLR의 NT-Xent 손실을 반환한다.

1. 두 행렬을 이어 붙여 $2N$개의 embedding을 만들고, 각 행을 단위 길이로 정규화한다.
2. 유사도 $s_{ab} = \mathbf{z}_a^\top\mathbf{z}_b / \tau$를 계산한다.
3. view $a$마다 자기 자신을 뺀 $2N - 1$개 후보 중 짝(다른 view의 같은 표본)을 고르는 cross-entropy를 구한다.

$$
\ell_a = -s_{a,\,p(a)} + \log\sum_{b \ne a} e^{s_{ab}}, \qquad L = \frac{1}{2N}\sum_{a=1}^{2N}\ell_a.
$$

$p(a)$는 $a$의 짝의 인덱스다(앞쪽 $N$개의 짝은 뒤쪽 $N$개, 그 반대도 마찬가지). $\tau$가 작으면 $s$가 커지므로 log-sum-exp를 안정적으로 계산한다(행마다 최댓값을 빼고 계산한 뒤 더한다).

테스트는 다음을 확인한다.

- 정의를 그대로 옮긴 반복문 구현과 값이 같다.
- 모든 embedding이 같으면(완전 붕괴) 손실은 $\log(2N - 1)$이다.
- 짝끼리만 같고 나머지는 서로 직교하면 $\tau$가 작을수록 손실이 0에 가까워진다.
- embedding의 크기를 바꿔도(행마다 양수를 곱해도) 값이 같다. 정규화 때문이다.
- `z1`과 `z2`를 바꿔도 값이 같다.
- $\tau = 0.01$처럼 작은 값에서도 `inf`나 `nan`이 나오지 않는다.
