# batch 공분산을 축 규약대로

센서 $B$개가 각각 $N$번 측정한 $D$차원 신호를 `X`, shape `(B, N, D)`로 받았다. 축의 의미는 batch(센서) · 샘플(측정 시점) · feature(측정 채널)다. 함수 `batch_cov(X)`를 완성해 **센서마다** 표본 공분산 행렬을 반환한다.

$$
\boldsymbol{\Sigma}_b = \frac{1}{N-1} \sum_{n=1}^{N} (\mathbf{x}_{bn} - \boldsymbol{\mu}_b)(\mathbf{x}_{bn} - \boldsymbol{\mu}_b)^\top,
\qquad
\boldsymbol{\mu}_b = \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_{bn}
$$

반환값의 shape은 `(B, D, D)`다.

테스트는 다음을 확인한다.

- 센서마다 `np.cov(X[b], rowvar=False)`와 같다.
- 평균은 **센서마다 따로**(샘플 축 `N`으로) 뺀다. 전체 batch의 평균을 빼면 센서마다 기준점이 다른 데이터에서 공분산이 부풀려진다.
- 결과는 대칭이고 고윳값이 음수가 아니다.

힌트: `X.mean(axis=?, keepdims=True)`로 중심화한 뒤, 합해서 사라질 축이 무엇인지 생각하며 einsum 문자열을 쓴다. 출력의 두 feature 축에는 서로 다른 글자가 필요하다.
