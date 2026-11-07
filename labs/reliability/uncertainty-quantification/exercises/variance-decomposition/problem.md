# 앙상블 예측의 분산 분해

앙상블 구성원 $m = 1, \dots, M$이 입력마다 가우시안 예측분포 $\mathcal{N}(\mu_m(x), \sigma_m^2(x))$를 낸다. 구성원을 고르게 섞은 혼합분포의 분산은 전분산 법칙으로

$$
\operatorname{Var}[y \mid x] = \underbrace{\frac{1}{M}\sum_m \sigma_m^2(x)}_{\text{aleatoric}} + \underbrace{\frac{1}{M}\sum_m \big(\mu_m(x) - \bar\mu(x)\big)^2}_{\text{epistemic}}, \qquad \bar\mu(x) = \frac{1}{M}\sum_m \mu_m(x)
$$

이다. 함수 `decompose(mu, var)`를 완성한다.

| 인자 | shape | 의미 |
|---|---|---|
| `mu` | `(M, N)` | 구성원 $m$이 입력 $n$에 대해 낸 평균 |
| `var` | `(M, N)` | 구성원 $m$이 입력 $n$에 대해 낸 분산(표준편차가 아니다) |

반환값은 `(aleatoric, epistemic, total)` 세 배열이고 각각 shape `(N,)`이다. epistemic 항은 구성원 수 $M$으로 나눈다(`ddof=0`). 이렇게 해야 `total`이 혼합분포의 분산과 정확히 같다.

테스트는 손으로 계산한 작은 예, shape, 그리고 혼합분포에서 직접 뽑은 표본의 분산과 `total`이 맞는지 확인한다.
