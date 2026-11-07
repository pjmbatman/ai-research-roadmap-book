# 두 가우시안 사이의 KL

1차원 가우시안 $p = \mathcal{N}(\mu_1, \sigma_1^2)$, $q = \mathcal{N}(\mu_2, \sigma_2^2)$ 사이의 KL divergence는 닫힌 형태로 쓸 수 있다.

$$
D_{\mathrm{KL}}(p \,\|\, q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12.
$$

`gaussian_kl(mu1, sigma1, mu2, sigma2)`를 완성해 $D_{\mathrm{KL}}(p \,\|\, q)$를 nat 단위로 반환한다. 인자는 스칼라이거나 같은 shape의 배열일 수 있다(배열이면 원소별로 계산한다). $\sigma$는 표준편차이고 양수다.

테스트는 다음을 확인한다.

- 수치적분으로 계산한 KL과 일치한다.
- $p = q$이면 0이다.
- 비대칭이다: 좁은 $p$와 넓은 $q$의 $D_{\mathrm{KL}}(p \,\|\, q)$와 $D_{\mathrm{KL}}(q \,\|\, p)$가 다르고, 이 경우 넓은 분포에서 좁은 분포로 재는 쪽이 더 크다.
- 단위 불변성: 평균과 표준편차를 모두 같은 상수배 하면 값이 변하지 않는다.

이어서 생각해 본다. $\sigma_1 = 0.1$, $\sigma_2 = 1$(평균은 같음)일 때 두 방향의 값을 계산하고, 어느 방향이 "$q$가 $p$를 덮지 못함"을 더 크게 벌하는지 본문의 forward·reverse 설명과 연결한다.
