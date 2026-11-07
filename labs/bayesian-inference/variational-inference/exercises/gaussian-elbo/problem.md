# 가우시안 변분분포의 ELBO와 KL

1차원 파라미터 $\theta$의 변분분포를 $q(\theta) = \mathcal{N}(m, s^2)$로 둔다. ELBO는

$$
\mathcal{L}(m, s) = \mathbb{E}_{q}\big[\log p(\mathcal{D}, \theta) - \log q(\theta)\big]
$$

이다. 기대값은 재매개변수화 $\theta = m + s\,\varepsilon$, $\varepsilon \sim \mathcal{N}(0, 1)$로 Monte Carlo 추정한다.

1. `elbo_mc(log_joint, m, s, eps)`: 표준정규 표본 배열 `eps`(리스트일 수도 있다)로 만든 $\theta$들에서 $\log p(\mathcal{D}, \theta) - \log q(\theta)$의 평균. `log_joint`는 $\theta$ 배열을 받아 같은 shape의 $\log p(\mathcal{D}, \theta)$ 배열을 돌려주는 함수다. $s \le 0$이면 `ValueError`.
2. `kl_gaussians(m1, s1, m2, s2)`: 두 1차원 가우시안 사이의 KL을 닫힌 꼴로 계산한다.

$$
\mathrm{KL}\big(\mathcal{N}(m_1, s_1^2)\,\|\,\mathcal{N}(m_2, s_2^2)\big) = \log\frac{s_2}{s_1} + \frac{s_1^2 + (m_1 - m_2)^2}{2 s_2^2} - \frac{1}{2}
$$

테스트는 사후분포가 가우시안이고 $\log p(\mathcal{D})$를 아는 경우에 $\mathcal{L} = \log p(\mathcal{D}) - \mathrm{KL}(q \,\|\, p(\theta \mid \mathcal{D}))$가 성립하는지 확인한다. $q$가 사후분포와 같으면 표본이 몇 개든 추정값이 정확히 $\log p(\mathcal{D})$가 된다(추정 분산 0)는 점도 확인한다. $\log p(\mathcal{D}, \theta) - \log q(\theta)$가 그때 어떤 함수가 되는지 생각해 본다.
