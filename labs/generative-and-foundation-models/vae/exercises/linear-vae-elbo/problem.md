# 선형 가우시안 VAE의 ELBO 두 항

잠재변수가 1차원인 선형 가우시안 VAE를 생각한다.

$$
p(z) = \mathcal{N}(0, 1), \qquad
p(\mathbf{x} \mid z) = \mathcal{N}\big(\mathbf{w} z + \mathbf{b},\ \sigma^2 \mathbf{I}_d\big), \qquad
q(z \mid \mathbf{x}) = \mathcal{N}(\mu_q, s_q^2).
$$

함수 `elbo_terms(x, w, b, sigma, mu_q, s_q)`를 완성해 ELBO의 두 항 `(rec, kl)`을 **닫힌 꼴로** 반환한다.

- 재구성 항 $\text{rec} = \mathbb{E}_{q}\big[\log p(\mathbf{x} \mid z)\big]$
- KL 항 $\text{kl} = \mathrm{KL}\big(q(z \mid \mathbf{x}) \,\|\, p(z)\big)$

ELBO는 `rec - kl`이다.

| 인자 | shape·의미 |
|---|---|
| `x`, `w`, `b` | `(d,)` |
| `sigma` | decoder 표준편차(양수 스칼라) |
| `mu_q`, `s_q` | encoder 평균과 표준편차(스칼라, `s_q > 0`) |

힌트:

- $\mathbf{r} = \mathbf{x} - \mathbf{b} - \mathbf{w}\mu_q$라 두면 $\mathbb{E}_q\lVert \mathbf{x} - \mathbf{b} - \mathbf{w}z \rVert^2 = \lVert\mathbf{r}\rVert^2 + \lVert\mathbf{w}\rVert^2 s_q^2$이다($z = \mu_q + s_q\varepsilon$를 대입해 전개한다).
- $d$차원 등방 가우시안의 로그밀도 상수항은 $-\tfrac{d}{2}\log(2\pi\sigma^2)$이다.
- 두 1차원 가우시안의 KL: $\mathrm{KL}\big(\mathcal{N}(\mu, s^2) \,\|\, \mathcal{N}(0, 1)\big) = \tfrac{1}{2}\big(\mu^2 + s^2 - 1 - \log s^2\big)$.

테스트는 Monte Carlo 추정과 비교하고, encoder가 정확한 사후분포일 때 ELBO가 정확한 $\log p(\mathbf{x})$와 같아지는지 확인한다.
