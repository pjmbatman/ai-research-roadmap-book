---
id: generative-and-foundation-models/gan-and-flows
title: "GAN과 Normalizing Flow"
title_en: "GANs and Normalizing Flows"
tier: advanced
status: draft
prerequisites: [generative-and-foundation-models/generative-modeling, calculus-and-autodiff/integration]
---

# GAN과 Normalizing Flow

19.3장의 VAE는 likelihood의 하한을 최대화했다. 이 장은 정반대 방향의 두 계열을 다룬다. GAN(generative adversarial network)은 likelihood를 **아예 쓰지 않는다.** 생성자가 만든 표본과 실제 데이터를 구별하는 판별자를 함께 학습시키고, 생성자는 판별자를 속이는 방향으로 움직인다. normalizing flow는 반대로 likelihood를 **정확히** 계산한다. 가역 변환으로 단순한 분포를 데이터 분포로 옮기고, 8.3장의 변수변환 공식으로 밀도를 계산한다.

두 계열은 19.1장의 표에서 양 끝에 있다. GAN은 표본이 날카롭지만 밀도를 줄 수 없고 학습이 불안정하다. flow는 밀도와 역변환을 주지만 구조에 강한 제약이 걸린다. 이 장은 각각의 원리를 수식과 작은 계산으로 확인하고, 같은 데이터에 둘을 학습해 무엇이 다른지 비교한다.

## 학습 목표

- 적대적 학습의 원리와 불안정성을 이해한다
- 가역 변환으로 정확한 likelihood를 계산한다

## GAN

**구성.** 생성자(generator) $G_\theta$는 잡음 $\mathbf{z} \sim p(\mathbf{z})$를 받아 $\mathbf{x} = G_\theta(\mathbf{z})$를 낸다. 이 표본들의 분포를 $p_g$라 하자. 19.1장의 암묵적 모델이므로 $p_g$의 밀도 식은 없다. 판별자(discriminator) $D_\psi(\mathbf{x}) \in (0, 1)$는 $\mathbf{x}$가 실제 데이터일 확률을 출력하는 분류기다. Goodfellow et al.(2014)은 두 신경망이 다음 값을 놓고 겨루는 minimax 문제를 제안했다.

$$
\min_\theta \max_\psi\ V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}\big[\log D(\mathbf{x})\big] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}\big[\log\big(1 - D(G(\mathbf{z}))\big)\big].
$$

판별자에게 $V$는 실제 데이터에 레이블 1, 생성 표본에 레이블 0을 붙인 이진 분류의 로그우도이고, 생성자는 그 값을 떨어뜨리려 한다. 실제 구현에서는 두 신경망을 번갈아 몇 step씩 경사법으로 갱신한다.

**divergence 해석.** 생성자를 고정하면 각 점 $\mathbf{x}$에서 $p_{\text{data}}\log D + p_g\log(1 - D)$를 최대화하는 판별자는

$$
D^\star(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}
$$

이다. 이를 대입하면

$$
V(D^\star, G) = -\log 4 + 2\,\mathrm{JS}(p_{\text{data}} \,\|\, p_g), \qquad
\mathrm{JS}(p \,\|\, q) = \tfrac{1}{2}\mathrm{KL}\big(p \,\|\, m\big) + \tfrac{1}{2}\mathrm{KL}\big(q \,\|\, m\big),\ m = \tfrac{p + q}{2}
$$

가 된다. 즉 판별자가 최적이라면 생성자는 Jensen–Shannon divergence(14.1장)를 최소화한다. JS는 0 이상 $\log 2$ 이하이고, $p_g = p_{\text{data}}$일 때만 0이다. 그때 $D^\star = 1/2$이 되어 판별자는 둘을 구별하지 못한다.

**코드로 확인.** 1차원 쌍봉 분포 $p_{\text{data}} = \tfrac{1}{2}\mathcal{N}(-2, 0.5^2) + \tfrac{1}{2}\mathcal{N}(2, 0.5^2)$과 생성자 분포 $\mathcal{N}(\mu, 0.5^2)$에 대해 $V(D^\star, G)$를 격자 적분으로 계산하고 $2\,\mathrm{JS} - \log 4$와 비교한다. 생성자 위치 $\mu$에 대한 기울기도 유한차분으로 구한다.

```python {exec="browser" session="gan"}
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

x = np.linspace(-12, 16, 7001)                     # (X,) 적분 격자
dx = x[1] - x[0]
log_pd = logsumexp([norm.logpdf(x, -2, 0.5), norm.logpdf(x, 2, 0.5)], axis=0) - np.log(2)
p_data = np.exp(log_pd)                            # 0.5 N(-2, 0.5^2) + 0.5 N(2, 0.5^2)

def value_at_optimal_d(log_pg):                    # V(D*, G) = E_data log D* + E_g log(1 - D*)
    log_den = np.logaddexp(log_pd, log_pg)         # D* = p_data / (p_data + p_g), 로그로 계산
    return np.sum(p_data * (log_pd - log_den) + np.exp(log_pg) * (log_pg - log_den)) * dx

def js(log_p, log_q):                              # 0.5 KL(p||m) + 0.5 KL(q||m), m = (p + q)/2
    log_m = np.logaddexp(log_p, log_q) - np.log(2)
    return 0.5 * np.sum(np.exp(log_p) * (log_p - log_m) + np.exp(log_q) * (log_q - log_m)) * dx

for mu in [0.0, 3.0, 4.5, 8.0]:
    v = value_at_optimal_d(norm.logpdf(x, mu, 0.5))
    grad = (value_at_optimal_d(norm.logpdf(x, mu + 0.01, 0.5))
            - value_at_optimal_d(norm.logpdf(x, mu - 0.01, 0.5))) / 0.02
    two_js = 2 * js(log_pd, norm.logpdf(x, mu, 0.5)) - np.log(4)
    print(f"생성자 N({mu:4.1f}, 0.5^2): V(D*,G) {round(v, 4) + 0.0:7.4f}  2JS - log4 {round(two_js, 4) + 0.0:7.4f}"
          f"  dV/dmu {round(grad, 3) + 0.0:6.3f}")
```

```output
생성자 N( 0.0, 0.5^2): V(D*,G) -0.1694  2JS - log4 -0.1694  dV/dmu  0.000
생성자 N( 3.0, 0.5^2): V(D*,G) -0.4961  2JS - log4 -0.4961  dV/dmu  0.620
생성자 N( 4.5, 0.5^2): V(D*,G) -0.0242  2JS - log4 -0.0242  dV/dmu  0.067
생성자 N( 8.0, 0.5^2): V(D*,G)  0.0000  2JS - log4  0.0000  dV/dmu  0.000
```

네 경우 모두 $V(D^\star, G) = 2\,\mathrm{JS} - \log 4$가 맞는다. 눈여겨볼 것은 마지막 열이다. 생성자가 데이터와 겹치는 $\mu = 3$에서는 기울기가 0.62로 데이터 쪽으로 미는 신호가 있지만, $\mu = 4.5$에서는 0.067로 약해지고 겹침이 사라진 $\mu = 8$에서는 0이다. 두 분포의 지지집합이 떨어지면 JS는 거리와 무관하게 $\log 2$로 포화되므로, **판별자가 완벽할수록 생성자는 어느 쪽으로 가야 할지 신호를 받지 못한다.** ($\mu = 0$에서 0인 것은 대칭 때문이다.) 고차원 데이터는 저차원 다양체 근처에 몰려 있어 학습 초기의 $p_g$와 지지집합이 겹치지 않는 경우가 흔하다. 그래서 원 논문도 실제 학습에서는 생성자가 $\log(1 - D(G(\mathbf{z})))$를 줄이는 대신 $\log D(G(\mathbf{z}))$를 키우도록 바꾼 **non-saturating** 손실을 권했다. 최적점은 같지만 판별자가 생성 표본을 쉽게 가려낼 때도 기울기가 살아 있다.

::quiz{id="optimal-discriminator"}

**divergence마다 최적해가 다르다.** 생성자가 가우시안 하나뿐이라 쌍봉 분포를 정확히 표현할 수 없다면, 어떤 divergence를 줄이느냐에 따라 "최선의 근사"가 달라진다.

```python {exec="browser" session="gan"}
def kl(log_p, log_q):
    return np.sum(np.exp(log_p) * (log_p - log_q)) * dx

mus, sds = np.linspace(-3, 3, 61), np.linspace(0.2, 3.0, 29)
for name, div in [("KL(data||g) (최대우도)", lambda lq: kl(log_pd, lq)),
                  ("JS", lambda lq: js(log_pd, lq)),
                  ("KL(g||data)", lambda lq: kl(lq, log_pd))]:
    vals = np.array([[div(norm.logpdf(x, m, s)) for s in sds] for m in mus])   # (61, 29)
    i, j = np.unravel_index(np.argmin(vals), vals.shape)
    print(f"{name:22s} 최적 생성자: |평균| {abs(mus[i]):.2f}, 표준편차 {sds[j]:.2f}  divergence {vals[i, j]:.3f}")
```

```output
KL(data||g) (최대우도)     최적 생성자: |평균| 0.00, 표준편차 2.10  divergence 0.724
JS                     최적 생성자: |평균| 0.00, 표준편차 2.00  divergence 0.201
KL(g||data)            최적 생성자: |평균| 2.00, 표준편차 0.50  divergence 0.693
```

최대우도(데이터를 앞에 둔 KL)와 JS는 두 모드를 모두 덮는 넓은 가우시안을 고르고, 순서를 뒤집은 KL은 모드 하나에 딱 맞는 가우시안을 고른다. KL$(p_g \,\|\, p_{\text{data}})$는 데이터가 없는 곳에 생성자가 질량을 두면 크게 벌점을 주므로 모드 사이를 비우는 쪽을 택한다(mode-seeking). 이 결과는 GAN의 목적함수인 JS 자체가 모드 하나만 고르는 해를 선호하지는 않는다는 것을 보여 준다. 그런데도 GAN은 실제로 모드를 자주 잃는다. 다음 절의 주제다.

## GAN의 어려움

**mode collapse.** 생성자가 데이터의 일부 모드만 만들어 내는 현상이다. 극단적인 경우 잡음 $\mathbf{z}$와 무관하게 거의 같은 표본을 낸다. 표본 하나하나는 그럴듯하므로 표본만 봐서는 잘 드러나지 않는다(19.1장의 '모드 절반' 모델).

**왜 일어나는가.** 생성자는 판별자의 **현재** 출력에서 국소적인 기울기만 받는다. 아래 코드는 앞 절의 쌍봉 데이터에 가우시안 생성자 $\mathcal{N}(\mu, s^2)$(평균과 표준편차 모두 학습)와 RBF 특징 위의 로지스틱 판별자를 번갈아 경사법으로 학습한다.

```python {exec="browser" session="gan"}
rng = np.random.default_rng(0)
C = np.linspace(-4, 4, 17)                         # 판별자 RBF 특징의 중심
phi = lambda v: np.exp(-(v[:, None] - C) ** 2 / (2 * 0.5**2))           # (n, 17)
sig = lambda z: 1 / (1 + np.exp(-z))
w, b, mu, s = np.zeros(17), 0.0, -0.3, 0.3        # 판별자 (w, b), 생성자 x = mu + s * eps
for t in range(601):
    xr = np.where(rng.random(128) < 0.5, -2, 2) + 0.5 * rng.normal(size=128)
    eps = rng.normal(size=128)
    xf = mu + s * eps                              # (128,) 생성 표본
    for _ in range(2):                             # 판별자: log D(real) + log(1 - D(fake)) 상승
        pr, pf = sig(phi(xr) @ w + b), sig(phi(xf) @ w + b)
        w += phi(xr).T @ (1 - pr) / 128 - phi(xf).T @ pf / 128
        b += np.mean(1 - pr) - np.mean(pf)
    pf = sig(phi(xf) @ w + b)                      # 생성자: log D(fake) 상승 (non-saturating)
    dlogit = (phi(xf) * (-(xf[:, None] - C) / 0.25)) @ w                  # d logit / d x
    mu += 0.05 * np.mean((1 - pf) * dlogit)
    s *= np.exp(0.05 * np.mean((1 - pf) * dlogit * eps) * s)              # log s 갱신
    if t % 150 == 0:
        print(f"step {t:3d}: 생성자 N({mu:5.2f}, {s:.2f}^2)")
print("판별자 D(x) at x = -2, 0, 2:", np.round(sig(phi(np.array([-2.0, 0.0, 2.0])) @ w + b), 3))
print(f"JS: 학습 결과 {js(log_pd, norm.logpdf(x, mu, s)):.3f}  vs  JS 최적 가우시안 {js(log_pd, norm.logpdf(x, 0, 2.0)):.3f}")
```

```output
step   0: 생성자 N(-0.30, 0.30^2)
step 150: 생성자 N(-2.03, 0.49^2)
step 300: 생성자 N(-2.03, 0.51^2)
step 450: 생성자 N(-2.01, 0.50^2)
step 600: 생성자 N(-2.00, 0.49^2)
판별자 D(x) at x = -2, 0, 2: [0.371 0.509 1.   ]
JS: 학습 결과 0.216  vs  JS 최적 가우시안 0.201
```

생성자는 출발점에서 가까운 왼쪽 모드로 가서 그 모드에 정확히 맞는 $\mathcal{N}(-2, 0.5^2)$에 머문다. 판별자는 오른쪽 모드가 진짜 데이터뿐이라는 것을 확실히 알지만($D(2) = 1.000$), 생성자 표본이 있는 곳에서의 기울기는 그 사실을 전하지 못한다. 왼쪽 모드에서 판별자 출력 0.371은 최적 판별자 값 $\tfrac{0.5}{0.5 + 1} = 1/3$에 가깝다. 이 해는 JS의 전역 최솟값(0.201, 넓은 가우시안)이 아니지만 그 값과의 차이가 0.015에 불과하다. JS는 "모드 하나를 완벽히"와 "두 모드를 흐릿하게"를 거의 구별하지 않고, 경사법은 가까운 쪽에 머문다. 신경망 생성자에서는 여기에 판별자와 생성자가 서로를 쫓으며 모드 사이를 옮겨 다니는 동역학이 더해진다.

**학습 불안정.** GAN 학습은 손실 하나를 최소화하는 것이 아니라 두 player의 **균형점**을 찾는 일이다. 번갈아 하는 경사법은 균형점 근처에서 수렴하지 않고 맴돌거나 발산할 수 있고, 판별자가 너무 강하면 앞 절의 기울기 소실이, 너무 약하면 의미 없는 신호가 생긴다. 그래서 학습률·갱신 비율·구조에 민감하다.

**안정화 기법.** 집필 시점에 널리 쓰이는 방법은 판별자를 매끄럽게 만드는 쪽이다. 판별자의 Lipschitz 상수를 제한하는 spectral normalization, 판별자 기울기의 크기에 벌점을 주는 gradient penalty, JS 대신 지지집합이 겹치지 않아도 거리를 반영하는 Wasserstein 거리를 쓰는 변형(WGAN)이 대표적이다. 모드 누락을 줄이려고 판별자에게 표본 하나가 아니라 mini-batch 통계를 보여 주는 방법도 쓰인다. 어느 것도 수렴을 보장하지는 않으므로, 학습 중 표본 기반 지표를 따로 기록하는 것이 기본이다.

**손실 값은 진행을 말해 주지 않는다.** 원 위의 모드 8개(19.1장의 데이터)에 작은 GAN을 학습하며 400 step마다 두 손실과 표본 지표를 기록한다. 이 셀은 PyTorch가 필요해 로컬 런타임에서 실행된다.

```python {exec="python"}
import torch
from torch import nn

torch.manual_seed(0)
torch.set_num_threads(1)                           # 작은 모델은 단일 스레드가 빠르다
ang = torch.arange(8) * 2 * torch.pi / 8
centers = 2.0 * torch.stack([ang.cos(), ang.sin()], 1)                  # (8, 2) 원 위의 모드
data = lambda n: centers[torch.randint(8, (n,))] + 0.15 * torch.randn(n, 2)
mlp = lambda out: nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, out))
G, D = mlp(2), mlp(1)                              # 생성자 z (2,) -> x (2,), 판별자 x -> logit
opt_g = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
ones, zeros = torch.ones(256, 1), torch.zeros(256, 1)
for step in range(1, 2001):
    xf = G(torch.randn(256, 2))                    # (256, 2)
    loss_d = bce(D(data(256)), ones) + bce(D(xf.detach()), zeros)
    opt_d.zero_grad(); loss_d.backward(); opt_d.step()
    loss_g = bce(D(xf), ones)                      # non-saturating 생성자 손실
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()
    if step % 400 == 0:
        with torch.no_grad():
            dist = torch.cdist(G(torch.randn(2000, 2)), centers)          # (2000, 8)
            near = dist.min(1).values < 0.3
            modes = (torch.bincount(dist.argmin(1)[near], minlength=8) > 20).sum()
        print(f"step {step:4d}  D 손실 {loss_d.item():.3f}  G 손실 {loss_g.item():.3f}"
              f"  모드 0.3 이내 {near.float().mean():.2f}  덮은 모드 {modes}/8")
```

```output
step  400  D 손실 1.329  G 손실 0.734  모드 0.3 이내 0.12  덮은 모드 5/8
step  800  D 손실 1.299  G 손실 0.776  모드 0.3 이내 0.52  덮은 모드 8/8
step 1200  D 손실 1.284  G 손실 0.791  모드 0.3 이내 0.69  덮은 모드 8/8
step 1600  D 손실 1.318  G 손실 0.791  모드 0.3 이내 0.76  덮은 모드 8/8
step 2000  D 손실 1.342  G 손실 0.805  모드 0.3 이내 0.66  덮은 모드 8/8
```

판별자 손실은 1.28–1.34, 생성자 손실은 0.73–0.81 사이에서 거의 움직이지 않는다. 같은 기간에 모드 근처 표본 비율은 0.12에서 0.76까지 올랐다가 2000 step에서 0.66으로 내려갔다(참 분포는 0.86). 두 player가 함께 좋아지면 상대가 강해진 만큼 각자의 손실이 그대로이기 때문이다. 이상적인 균형점의 판별자 손실은 $2\log 2 \approx 1.386$이고 생성자 손실(non-saturating)은 $\log 2 \approx 0.693$이다. 손실이 이 값 근처라는 사실만으로는 생성자가 좋다는 것도 나쁘다는 것도 알 수 없다.

:::note[분야 예시]
입자물리 실험에서는 검출기 반응을 정밀하게 시뮬레이션하는 데 계산 자원의 큰 부분이 들어간다. 그래서 시뮬레이션 출력(예: 칼로리미터의 에너지 분포)을 흉내 내는 빠른 생성모델, 즉 대리 시뮬레이터(fast simulation)로 GAN과 flow 같은 생성모델을 쓰는 연구가 이어져 왔다. 이런 응용에서는 표본이 그럴듯해 보이는 것보다 **물리량 분포의 꼬리**와 입력 조건(입자 에너지·각도)에 따른 변화를 정확히 재현하는지가 중요하고, 이 장의 mode collapse와 손실 해석 문제가 그대로 검증 항목이 된다.
:::

## Normalizing flow

**왜 필요한가.** GAN의 생성자 $\mathbf{x} = G(\mathbf{z})$가 **가역**이고 $\mathbf{z}$와 $\mathbf{x}$의 차원이 같다면, $\mathbf{x}$의 밀도를 정확히 계산할 수 있다. likelihood가 있으면 최대우도로 안정적으로 학습할 수 있고, 19.1장의 세 능력을 모두 갖는다.

**정의.** 기저 분포 $p_Z$(보통 $\mathcal{N}(\mathbf{0}, \mathbf{I})$)와 가역이고 미분 가능한 변환 $f_\theta: \mathbb{R}^d \to \mathbb{R}^d$로 $\mathbf{x} = f_\theta(\mathbf{z})$를 정의한다. 8.3장의 확률밀도 변환 공식에서

$$
\log p_X(\mathbf{x}) = \log p_Z\big(f_\theta^{-1}(\mathbf{x})\big) - \log\Big\lvert \det \frac{\partial f_\theta}{\partial \mathbf{z}}\Big\rvert_{\mathbf{z} = f_\theta^{-1}(\mathbf{x})}
$$

이다. 정변환이 공간을 늘리는 곳에서는 같은 확률 질량이 넓은 부피에 퍼지므로 밀도가 그만큼 낮아진다. Jacobian 행렬식이 그 부피 배율이다. 변환을 $f = f_K \circ \cdots \circ f_1$처럼 쌓으면 log-det은 층별 값의 합이 된다. 이름의 "normalizing"은 데이터를 역변환하면 표준정규분포로 "정규화"된다는 뜻이고, "flow"는 이 변환들의 연쇄를 가리킨다(Papamakarios et al., 2021).

**학습과 생성.** 학습은 데이터 $\mathbf{x}$를 **역방향**으로 보내 $\mathbf{z}$와 log-det을 계산하고 위 식을 최대화한다. 생성은 $\mathbf{z} \sim p_Z$를 **정방향**으로 보낸다. 그래서 flow 층은 두 방향이 모두 싸야 하고, 행렬식도 싸야 한다. 일반적인 $d \times d$ 행렬식은 $O(d^3)$이므로 고차원에서 매 step 계산할 수 없다.

::quiz{id="flow-log-density"}

## flow 구조

**coupling layer.** 좌표를 두 묶음 $(\mathbf{x}_a, \mathbf{x}_b)$로 나누고, 한 묶음은 그대로 둔 채 다른 묶음만 그것에 의존하는 아핀 변환으로 바꾼다(Dinh, Sohl-Dickstein, Bengio, 2017).

$$
\mathbf{y}_a = \mathbf{x}_a, \qquad \mathbf{y}_b = \mathbf{x}_b \odot \exp\big(\mathbf{s}(\mathbf{x}_a)\big) + \mathbf{t}(\mathbf{x}_a).
$$

$\mathbf{s}$와 $\mathbf{t}$는 아무 신경망이어도 된다. 역변환은 $\mathbf{x}_b = (\mathbf{y}_b - \mathbf{t}(\mathbf{y}_a)) \odot \exp(-\mathbf{s}(\mathbf{y}_a))$로 닫힌 꼴이고 $\mathbf{s}, \mathbf{t}$를 뒤집을 필요가 없다. Jacobian은 블록 삼각행렬이고 대각 블록이 $\mathbf{I}$와 $\mathrm{diag}(\exp\mathbf{s})$이므로 $\log\lvert\det\rvert = \sum_j s_j(\mathbf{x}_a)$로 $O(d)$다. 한 층은 $\mathbf{x}_a$를 바꾸지 못하므로 층마다 나누는 방식을 번갈아 쓴다. 8.3장 실습의 바나나 변환이 이 층 하나였다.

**코드로 확인.** 2차원에서 두 coupling 층을 번갈아 쌓는다. 첫 층은 $z_1$을 고정하고 $z_2$를, 둘째 층은 그 결과의 두 번째 좌표를 고정하고 첫 번째 좌표를 바꾼다. 역변환 오차, 공식으로 계산한 log-det과 수치 Jacobian의 비교, 격자로 적분한 전체 질량을 확인한다.

```python {exec="browser" session="flow"}
import numpy as np

def s_net(u): return 0.6 * np.tanh(u)              # 고정된 '신경망' (학습 없이 구조만 확인)
def t_net(u): return 1.2 * np.sin(1.5 * u)

def forward(z):                                    # z (n, 2) -> x (n, 2), log|det dx/dz|
    x1, x2 = z[:, 0], z[:, 1] * np.exp(s_net(z[:, 0])) + t_net(z[:, 0])   # 층 1: x1 고정
    y2, y1 = x2, x1 * np.exp(s_net(x2)) + t_net(x2)                      # 층 2: x2 고정
    return np.stack([y1, y2], 1), s_net(z[:, 0]) + s_net(x2)

def inverse(y):                                    # 층 순서를 거꾸로, 각 층은 닫힌 꼴
    x2 = y[:, 1]
    x1 = (y[:, 0] - t_net(x2)) * np.exp(-s_net(x2))
    z1 = x1
    return np.stack([z1, (x2 - t_net(z1)) * np.exp(-s_net(z1))], 1)

def log_prob(y):                                   # log p(y) = log N(z; 0, I) - log|det dy/dz|
    z = inverse(y)
    return -0.5 * (z**2).sum(1) - np.log(2 * np.pi) - forward(z)[1]

z = np.random.default_rng(1).normal(size=(5, 2))
y, logdet = forward(z)
J = np.stack([(forward(z + e)[0] - forward(z - e)[0]) / 2e-6 for e in np.eye(2) * 1e-6], 2)  # (5, 2, 2)
print("역변환 오차 < 1e-12:", np.abs(inverse(y) - z).max() < 1e-12)
print("log|det J| 공식 - 수치 Jacobian, 최대 차이 < 1e-6:", np.abs(logdet - np.log(np.abs(np.linalg.det(J)))).max() < 1e-6)
g = np.linspace(-10, 10, 1001)
Y = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)  # (1001^2, 2) 격자
print(f"격자로 적분한 전체 질량: {np.exp(log_prob(Y)).sum() * (g[1] - g[0]) ** 2:.4f}")
```

```output
역변환 오차 < 1e-12: True
log|det J| 공식 - 수치 Jacobian, 최대 차이 < 1e-6: True
격자로 적분한 전체 질량: 1.0000
```

역변환은 반올림 오차 수준으로 정확하고, 층별 $s$의 합이 수치 Jacobian의 log-det과 같으며, 공식으로 계산한 밀도는 적분하면 1이다. 학습에서는 $s, t$ 자리에 신경망을 넣고 $-\log p(\mathbf{x})$의 평균을 손실로 쓰면 된다.

**autoregressive flow.** 각 좌표의 아핀 변환이 그 앞의 모든 좌표에 의존하게 하면 $y_i = x_i \exp(s_i(\mathbf{x}_{<i})) + t_i(\mathbf{x}_{<i})$가 된다. Jacobian은 삼각행렬이라 log-det이 여전히 싸고, coupling보다 표현력이 크다. 대신 한쪽 방향은 19.2장의 자기회귀 생성처럼 좌표 수만큼 순차 계산이 필요하다. 어느 방향을 빠르게 만들지에 따라 밀도 계산이 빠른 구조와 표본 생성이 빠른 구조로 나뉜다. 실제로 자기회귀 모델 자체가 가우시안 head를 쓰면 이런 flow의 한 층이다.

**연속 flow.** 층을 무한히 잘게 쪼개면 변환은 ODE $d\mathbf{x}/dt = \mathbf{v}_\theta(\mathbf{x}, t)$의 해가 되고, log-det은 시간에 대한 적분 $-\int \mathrm{tr}(\partial\mathbf{v}_\theta/\partial\mathbf{x})\,dt$로 바뀐다. 구조 제약 없이 아무 신경망 $\mathbf{v}_\theta$나 쓸 수 있지만, 학습과 likelihood 계산에 ODE 적분이 필요해 비싸다. 19.6장의 flow matching은 이 ODE를 적분 없이 학습하는 방법이다.

**한계.** flow는 차원을 줄일 수 없다. 잠재 공간이 데이터와 같은 $d$차원이어야 하므로 저차원 표현을 얻는 용도에는 맞지 않고, 데이터가 저차원 다양체 위에 있으면 그 다양체 밖으로 밀도를 퍼뜨려야 해서 모델링이 어렵다. 또 가역 연속 변환은 공간의 연결 구조를 바꿀 수 없으므로, 하나로 연결된 가우시안을 떨어진 모드 여러 개로 옮기려면 모드 사이에 얇은 다리 모양의 밀도가 남는다. 실습에서 이 현상을 숫자로 본다.

:::note[분야 예시]
통계역학에서 Boltzmann 분포 $p(\mathbf{x}) \propto \exp(-U(\mathbf{x})/k_BT)$의 독립 표본을 얻는 일에 flow가 쓰인다. flow로 뽑은 표본 $\mathbf{x}$마다 정확한 $\log p_\theta(\mathbf{x})$를 알기 때문에, 중요도 가중치 $w \propto \exp(-U(\mathbf{x})/k_BT)/p_\theta(\mathbf{x})$(11.6장)로 생성모델의 오차를 보정해 자유에너지 같은 열역학량을 편향 없이 추정할 수 있다. GAN이나 VAE로는 이 가중치를 계산할 수 없다. likelihood가 정확해야만 가능한 응용이다.
:::

## 실습

같은 2차원 데이터(원 위의 모드 8개, 모드 표준편차 0.15)에 GAN과 affine coupling flow(8층, 층마다 은닉층 둘인 MLP)를 학습해 비교한다. 코드는 `labs/generative-and-foundation-models/gan-and-flows/gan_vs_flow.py`이고 CPU에서 1분 안팎 걸린다. 지표는 19.1장과 같은 표본 지표(모드 0.3 이내 비율, 덮은 모드 수, 모드별 비율)에 두 가지를 더했다. 표본 공백은 반지름 0.1 안에 생성 표본이 하나도 없는 test 점의 비율이고, KDE NLL은 두 모델의 표본에 같은 커널 밀도 추정을 적용한 test NLL이다. 집필 시점의 실행 결과는 다음과 같다.

```output
모델    모드 0.3 이내  덮은 모드  모드별 비율(최소–최대)  표본 공백  test NLL(정확)  test NLL(KDE)
참 분포          0.87       8/8       0.117–0.132      0.007         1.097       1.166
GAN           0.75       8/8       0.118–0.138      0.216         계산 불가       2.089
flow          0.61       8/8       0.118–0.135      0.006         1.456       1.502
(표본 공백: 0.1 이내에 표본이 하나도 없는 test 점의 비율. KDE: 표본 4000개, 커널 폭 0.1. 참 분포의 모드별 비율은 0.125)
```

결과를 읽는 법은 네 가지다.

- **표본만 보면 GAN이 낫다.** 모드 근처 표본 비율이 0.75로 flow(0.61)보다 높고, 이 실행에서는 모드 8개를 고른 비율로 모두 덮어 mode collapse도 없다.
- **분포로 보면 flow가 낫다.** GAN 표본은 모드 안을 고르게 채우지 못해 test 점의 22%는 근처에 GAN 표본이 하나도 없다. 그 결과 같은 KDE로 잰 test NLL이 GAN 2.09, flow 1.50이다. flow 표본이 모드 밖으로 더 새는 것은 앞 절의 한계(연결된 가우시안을 떨어진 모드들로 옮겨야 함) 때문이다.
- **flow만 정확한 likelihood를 준다.** flow의 정확한 test NLL(1.456)과 KDE 추정(1.502)이 가까워, 이 2차원 예에서는 KDE가 쓸 만한 근사라는 것도 확인할 수 있다. GAN은 KDE 외에 likelihood를 얻을 방법이 없고, 19.1장에서 보았듯 그 값은 커널 폭과 표본 수에 따라 움직인다.
- **참 분포의 정확한 NLL(1.097)도 KDE로는 1.166이다.** 같은 추정법을 쓴 열끼리만 비교해야 한다.

이어서 연습문제에서 coupling flow 한 층의 log-likelihood를 변수변환 공식으로 직접 구현한다.

::exercise{id="coupling-log-prob"}

## 실패 사례

- **GAN의 손실 값으로 학습 진행을 판단하는 것.** 셀 4에서 표본 품질이 0.12에서 0.76까지 바뀌는 동안 두 손실은 거의 그대로였다. 판별자 손실이 0으로 떨어지면 기울기 소실의 신호일 수는 있지만, 손실이 안정적이라는 것은 아무것도 보장하지 않는다. 학습 중에는 고정한 잡음 $\mathbf{z}$로 만든 표본과 표본 기반 지표(19.7장)를 일정 간격으로 기록하고, 그 지표로 checkpoint를 고른다.
- **표본이 좋아 보이면 분포도 맞는다고 보는 것.** 실습의 GAN은 표본 지표가 flow보다 좋았지만 test 점의 22% 근처를 비워 두었다. 모드 커버리지와 함께 데이터 쪽에서 본 지표(표본 공백, recall 계열 지표)를 본다.
- **flow의 log-det 부호나 방향을 뒤집는 것.** 정변환 Jacobian의 log-det은 **빼고**, 역변환 Jacobian의 log-det은 **더한다**. 부호를 틀리면 손실은 잘 줄어드는데 밀도는 적분해서 1이 되지 않는다. 구현한 뒤에는 2차원 격자 적분으로 전체 질량이 1인지 확인한다(셀 5).
- **이산 데이터에 연속 flow를 그대로 맞추는 것.** 정수 픽셀값처럼 이산적인 데이터에 연속 밀도를 맞추면 모델이 각 정수 점에 무한히 뾰족한 밀도를 두어 likelihood를 끝없이 키울 수 있다. 균등 잡음을 더하는 dequantization으로 데이터를 연속으로 만든 뒤 학습하고, 결과를 이산 확률(bits per dimension 등)로 보고할 때는 그 보정을 밝힌다.

## 통과 기준

- 최적 판별자를 대입해 GAN의 목적함수가 JS divergence가 됨을 유도하고, 지지집합이 떨어질 때 기울기가 사라지는 이유를 설명한다.
- flow의 log-likelihood 계산을 변수변환 공식으로 설명하고, coupling 층의 역변환과 log-det을 구현해 전체 질량이 1임을 확인한다.
- 같은 데이터에서 GAN과 flow를 표본 지표와 likelihood로 비교하는 표를 만들고 해석한다.
- 연습문제 `coupling-log-prob`의 테스트를 통과한다.

## 참고 문헌

- Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville, Bengio (2014). Generative adversarial nets. *NeurIPS 27*, 2672–2680. arXiv:1406.2661. — GAN의 minimax 목적함수, 최적 판별자와 JS divergence
- Dinh, Sohl-Dickstein, Bengio (2017). Density estimation using Real NVP. *ICLR 2017*. arXiv:1605.08803. — affine coupling 층으로 정확한 log-likelihood·표본·잠재 추론을 주는 flow
- Papamakarios, Nalisnick, Rezende, Mohamed, Lakshminarayanan (2021). Normalizing flows for probabilistic modeling and inference. *JMLR*, 22(57), 1–64. — normalizing flow의 개관
