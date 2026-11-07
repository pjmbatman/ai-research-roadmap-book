---
id: scientific-ml/inverse-problems-and-uq
title: "역문제와 불확실성 정량화"
title_en: "Inverse Problems and Uncertainty Quantification"
tier: advanced
status: draft
prerequisites: [scientific-ml/system-identification-and-estimation, bayesian-inference/mcmc]
---

# 역문제와 불확실성 정량화

시뮬레이터는 원인에서 결과를 계산한다. 초기 온도 분포에서 나중의 온도를, 지하의 물성에서 지표의 지진파 기록을, 반응 속도 상수에서 농도 곡선을 낸다. 과학에서 정말 알고 싶은 것은 흔히 그 반대 방향이다. 측정한 결과에서 원인을 거꾸로 추정하는 이 문제를 **역문제**(inverse problem)라 한다. 역문제는 대부분 측정 잡음을 엄청나게 키우거나, 같은 측정을 설명하는 원인이 여럿이다. 그래서 "답 하나"를 내는 것으로는 부족하고, 답이 **얼마나 불확실한지**를 함께 내야 한다.

이 장은 역문제가 왜 불안정한지와 정규화로 안정시키는 방법에서 시작해, 같은 문제를 베이지안 추론으로 다시 쓰고, 어떤 입력이 출력의 불확실성을 만드는지(민감도 분석)와 파라미터의 불확실성이 예측에 어떻게 퍼지는지(불확실성 전파)로 나아간다. 딥러닝 모델 자체의 불확실성(앙상블, MC dropout 등)은 21.1장에서 다루고, 여기서는 물리 모델·시뮬레이터의 파라미터와 입력의 불확실성에 집중한다.

## 학습 목표

- 불안정한 역문제를 정규화로 안정하게 푼다
- 역문제를 베이지안으로 정의하고 사후분포를 계산한다
- 민감도 분석과 불확실성 전파를 수행한다

## 역문제

**왜 필요한가.** 9.2장에서 본 것처럼 역문제는 대부분 조건이 나쁘다. 앞방향 과정(확산, 흐림, 적분)이 세부 구조를 지우기 때문에, 그것을 거꾸로 되살리려면 측정 잡음까지 함께 증폭하게 된다. 이 불안정성을 이해하지 않고 역문제를 풀면 잡음을 "복원"한 결과를 얻는다.

**정의.** 앞방향 모델(forward model) $G$와 측정 잡음 $\boldsymbol{\varepsilon}$으로 측정이 $\mathbf{y} = G(\mathbf{x}) + \boldsymbol{\varepsilon}$일 때, $\mathbf{y}$에서 $\mathbf{x}$를 추정하는 것이 역문제다. Hadamard는 해가 존재하고, 유일하고, 데이터에 연속적으로 의존할 때(작은 데이터 변화가 작은 해 변화를 낳을 때) 문제가 **잘 놓였다**(well-posed)고 했다. 셋 중 하나라도 깨지면 **잘못 놓인**(ill-posed) 문제다. 이산화한 선형 역문제 $\mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}$에서는 세 번째 조건, 안정성이 문제가 된다.

**수식.** $\mathbf{A}$의 SVD $\mathbf{A} = \sum_i s_i \mathbf{u}_i\mathbf{v}_i^\top$(7.6장)로 최소제곱 해를 쓰면

$$
\hat{\mathbf{x}} = \sum_i \frac{\mathbf{u}_i^\top\mathbf{y}}{s_i}\,\mathbf{v}_i = \mathbf{x} + \sum_i \frac{\mathbf{u}_i^\top\boldsymbol{\varepsilon}}{s_i}\,\mathbf{v}_i
$$

이다. 잡음의 각 성분이 $1/s_i$배로 증폭된다. 특이값이 $10^{-10}$이면 잡음 0.01이 $10^8$이 된다. **Tikhonov 정규화**는 해의 크기에 벌점을 준다.

$$
\hat{\mathbf{x}}_\lambda = \arg\min_{\mathbf{x}} \lVert \mathbf{A}\mathbf{x} - \mathbf{y} \rVert^2 + \lambda \lVert \mathbf{x} \rVert^2 = (\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})^{-1}\mathbf{A}^\top\mathbf{y} = \sum_i \underbrace{\frac{s_i^2}{s_i^2 + \lambda}}_{f_i} \frac{\mathbf{u}_i^\top\mathbf{y}}{s_i}\,\mathbf{v}_i.
$$

필터 인자 $f_i$는 $s_i^2 \gg \lambda$인 성분을 거의 그대로 두고($f_i \approx 1$), $s_i^2 \ll \lambda$인 성분을 누른다($f_i \approx s_i^2/\lambda$). 16.2장의 ridge 회귀와 같은 식이다. 작은 특이값 방향을 아예 버리는 절단 SVD(truncated SVD)도 같은 생각이다. 해가 매끄러워야 한다는 지식이 있으면 $\lVert \mathbf{x} \rVert^2$ 대신 미분 연산자 $\mathbf{L}$로 $\lVert \mathbf{L}\mathbf{x} \rVert^2$에 벌점을 준다.

**코드.** 80개 격자 위의 신호(상자 하나와 가우시안 봉우리 하나)를 폭 0.03의 가우시안 커널로 흐리게 하고 잡음 0.01을 더한 측정에서 원래 신호를 복원한다.

```python {exec="browser" session="inverse"}
import numpy as np

n = 80
s = np.linspace(0, 1, n)                                     # (n,) 위치
A = np.exp(-0.5 * (s[:, None] - s[None, :])**2 / 0.03**2)
A /= A.sum(axis=1, keepdims=True)                            # (n, n) 흐림 연산자: 이웃의 가중 평균
x_true = (np.abs(s - 0.3) < 0.1) + 0.5 * np.exp(-0.5 * (s - 0.7)**2 / 0.05**2)   # 참 신호 (n,)
rng = np.random.default_rng(0)
sigma = 0.01                                                 # 측정 잡음 표준편차
y = A @ x_true + sigma * rng.normal(size=n)                  # 측정 (n,)

def tikhonov(A, y, lam):                                     # argmin ||A x - y||^2 + lam ||x||^2
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)

sv = np.linalg.svd(A, compute_uv=False)
print(f"특이값 최대 {sv[0]:.2f}, 1e-6 미만인 것 {np.sum(sv < 1e-6)}개 / {n}")
print("정규화 없이 풀면 오차 > 1e3:", np.linalg.norm(np.linalg.lstsq(A, y, rcond=None)[0] - x_true) > 1e3)
for lam in [1e-6, 1e-4, 1e-2, 1e0]:
    x = tikhonov(A, y, lam)
    print(f"lam = {lam:.0e}: 잔차 {np.linalg.norm(A @ x - y):.3f}  해 노름 {np.linalg.norm(x):5.2f}"
          f"  오차 {np.linalg.norm(x - x_true):5.2f}")
```

```output
특이값 최대 1.00, 1e-6 미만인 것 22개 / 80
정규화 없이 풀면 오차 > 1e3: True
lam = 1e-06: 잔차 0.055  해 노름  9.77  오차  8.94
lam = 1e-04: 잔차 0.060  해 노름  4.42  오차  1.72
lam = 1e-02: 잔차 0.086  해 노름  4.07  오차  0.80
lam = 1e+00: 잔차 2.030  해 노름  1.92  오차  2.42
```

특이값 80개 중 22개가 $10^{-6}$보다 작다. 정규화 없이 풀면 오차가 $10^3$을 넘는다(값 자체는 반올림 잡음에 따라 달라지므로 판정만 찍었다). $\lambda$가 커질수록 잔차는 커지고 해의 노름은 작아진다. 오차는 그 사이 어딘가에서 가장 작다. $\lambda = 10^{-6}$에서는 잡음이 증폭되어 해가 출렁이고(오차 8.94), $\lambda = 1$에서는 해가 지나치게 눌려 측정조차 설명하지 못한다(잔차 2.030).

::quiz{id="ill-posed-noise-amplification"}

**$\lambda$ 고르기.** 참값을 모르는 실제 문제에서는 오차를 직접 볼 수 없다. 널리 쓰는 기준이 둘 있다.

- **L-curve**: $\lambda$를 바꿔 가며 $(\log\lVert \mathbf{A}\mathbf{x}_\lambda - \mathbf{y} \rVert, \log\lVert \mathbf{x}_\lambda \rVert)$를 그리면 L자 모양이 된다. 잔차를 거의 늘리지 않고 해 노름을 줄이는 구간과, 해 노름은 거의 그대로인데 잔차만 늘어나는 구간이 만나는 **모서리**(곡률 최대점)를 고른다(Hansen, O'Leary, 1993).
- **불일치 원리**(discrepancy principle): 잡음 수준 $\sigma$를 알면, 잔차가 잡음 크기 $\sqrt{n}\,\sigma$와 같아지는 $\lambda$를 고른다. 잡음보다 더 작은 잔차를 원하는 것은 잡음을 맞추겠다는 뜻이기 때문이다.

```python {exec="browser" session="inverse"}
lams = np.logspace(-8, 1, 91)
xs = [tikhonov(A, y, l) for l in lams]
res = np.array([np.linalg.norm(A @ x - y) for x in xs])
sol = np.array([np.linalg.norm(x) for x in xs])
err = np.array([np.linalg.norm(x - x_true) for x in xs])
r, e = np.log(res), np.log(sol)                              # L-curve: (log 잔차, log 해 노름)
dr, de = np.gradient(r), np.gradient(e)
curv = (dr * np.gradient(de) - de * np.gradient(dr)) / (dr**2 + de**2)**1.5
i_corner = np.argmax(curv[5:-5]) + 5                         # 양 끝의 수치 잡음 제외
i_disc = np.argmin(np.abs(res - np.sqrt(n) * sigma))         # 불일치 원리: 잔차 ≈ 잡음 크기
for name, i in [("L-curve 모서리  ", i_corner), ("불일치 원리     ", i_disc), ("오차 최소(참값 필요)", np.argmin(err))]:
    print(f"{name}  lam = {lams[i]:.1e}  오차 {err[i]:.2f}")
```

```output
L-curve 모서리    lam = 4.0e-04  오차 0.99
불일치 원리       lam = 1.0e-02  오차 0.80
오차 최소(참값 필요)  lam = 3.2e-03  오차 0.77
```

두 기준 모두 참값을 쓰지 않고도 최적에 가까운 $\lambda$를 찾았다. 이 예에서는 잡음 수준을 정확히 아는 불일치 원리가 더 가까웠고, L-curve 모서리는 조금 작은 $\lambda$를 골랐다. L-curve는 모서리가 뚜렷하지 않거나 곡률 추정이 격자에 민감할 수 있다. 잡음 수준을 모르면 일반화 교차검증(GCV)도 쓴다. 어느 기준이든 하나만 믿기보다 두세 개가 비슷한 범위를 가리키는지 확인한다.

:::note[분야 예시]
- **의료 영상**: CT는 여러 각도의 투영(선적분)에서 단면 영상을 재구성한다. 투영 수가 적거나 잡음이 크면 재구성에 줄무늬가 생기고, 정규화(전체 변동 벌점 등)의 강도가 영상의 선명도와 잡음 사이의 절충을 정한다.
- **지구물리**: 지표에서 잰 지진파 도달 시간이나 중력 이상으로 지하의 속도·밀도 분포를 추정한다. 깊은 곳일수록 측정의 민감도가 낮아 정규화가 해를 지배하므로, 깊은 구조의 해석에는 특히 조심한다.
:::

::widget{name="l-curve"}

## 베이지안 역문제

**왜 필요한가.** 정규화는 해를 하나 주지만, 그 해가 얼마나 믿을 만한지는 말하지 않는다. 베이지안 관점은 역문제의 답을 **사후분포**로 정의해, 해와 불확실성을 함께 준다(Stuart, 2010).

**정의.** 사전분포 $p(\mathbf{x})$(측정 전에 아는 것), 우도 $p(\mathbf{y} \mid \mathbf{x})$(앞방향 모델과 잡음 모델), 사후분포

$$
p(\mathbf{x} \mid \mathbf{y}) \propto p(\mathbf{y} \mid \mathbf{x})\,p(\mathbf{x}).
$$

선형 가우시안 경우 $\mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$, $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$에서 사후분포는 가우시안이고 닫힌 꼴이다(11.4장, 13.2장).

$$
\boldsymbol{\Sigma}_{\text{post}} = \left(\frac{\mathbf{A}^\top\mathbf{A}}{\sigma^2} + \frac{\mathbf{I}}{\tau^2}\right)^{-1}, \qquad
\boldsymbol{\mu}_{\text{post}} = \boldsymbol{\Sigma}_{\text{post}}\frac{\mathbf{A}^\top\mathbf{y}}{\sigma^2}.
$$

정리하면 $\boldsymbol{\mu}_{\text{post}} = (\mathbf{A}^\top\mathbf{A} + (\sigma^2/\tau^2)\mathbf{I})^{-1}\mathbf{A}^\top\mathbf{y}$로, **사후 평균(=MAP)은 $\lambda = \sigma^2/\tau^2$인 Tikhonov 해**다. 정규화 강도는 "잡음 분산 대 사전분산의 비"라는 해석을 얻고, 정규화 항은 사전분포가 된다.

```python {exec="browser" session="inverse"}
tau = 0.5                                                    # 사전분포 x ~ N(0, tau^2 I)
Sigma_post = np.linalg.inv(A.T @ A / sigma**2 + np.eye(n) / tau**2)   # (n, n) 사후 공분산
mu_post = Sigma_post @ A.T @ y / sigma**2                    # (n,) 사후 평균
sd_post = np.sqrt(np.diag(Sigma_post))
print("사후 평균 = Tikhonov(lam = sigma^2 / tau^2):", np.allclose(mu_post, tikhonov(A, y, sigma**2 / tau**2)))
print(f"점마다의 사후 표준편차 {sd_post.min():.3f} – {sd_post.max():.3f} (사전 {tau})")
print(f"95% 구간이 참값을 포함한 위치 비율 {np.mean(np.abs(mu_post - x_true) < 1.96 * sd_post):.2f}")
w = (np.abs(s - 0.3) < 0.1) / np.sum(np.abs(s - 0.3) < 0.1)   # 구간 [0.2, 0.4]의 평균을 내는 가중치 (n,)
print(f"구간 평균: 사후 {w @ mu_post:.3f} ± {np.sqrt(w @ Sigma_post @ w):.3f}  참값 {w @ x_true:.3f}")
```

```output
사후 평균 = Tikhonov(lam = sigma^2 / tau^2): True
점마다의 사후 표준편차 0.258 – 0.396 (사전 0.5)
95% 구간이 참값을 포함한 위치 비율 1.00
구간 평균: 사후 0.967 ± 0.022  참값 1.000
```

$\tau = 0.5$는 $\lambda = 4 \times 10^{-4}$에 해당한다(cell 2의 L-curve 모서리와 같은 값이다). 결과에서 두 가지를 읽는다.

- **점마다의 불확실성은 크다.** 사후 표준편차가 0.26–0.40으로 사전 표준편차 0.5에서 크게 줄지 않았다. 흐림은 이웃 점을 섞으므로, 데이터는 개별 점의 값을 거의 정하지 못한다. 95% 구간은 모든 위치에서 참값을 포함하지만, 그만큼 넓다.
- **평균 같은 매끄러운 양은 잘 정해진다.** 구간 [0.2, 0.4]의 평균은 $0.967 \pm 0.022$로, 점마다의 표준편차보다 10배 이상 정확하다. 사후 공분산의 비대각 성분(인접 점끼리의 음의 상관)이 평균에서 상쇄되기 때문이다.

**어떤 양을 묻느냐에 따라 불확실성이 다르다.** 역문제의 결과를 보고할 때는 점마다의 구간만 그리지 말고, 연구 질문이 묻는 양(평균, 적분, 최댓값의 위치)의 사후분포를 계산한다. 점마다의 구간을 이어 그린 띠는 사후분포에서 뽑은 개별 해가 어떤 모양인지 보여 주지 않는다.

**비선형 문제와 MCMC.** 앞방향 모델이 비선형이면 사후분포는 닫힌 꼴이 아니다. 파라미터가 몇 개라면 13.5장의 MCMC로 표본을 뽑는다. 효소 반응 속도의 Michaelis–Menten 모델 $v = v_{\max}S/(K_m + S)$에서 $v_{\max}$와 $K_m$을 측정 10개로 추정한다. 기질 농도 $S$가 $K_m$보다 훨씬 작은 범위만 재면 $v \approx (v_{\max}/K_m)S$라서 비율만 보인다(24.6장의 실용적 식별 불가능). 로그 파라미터에 넓은 가우시안 사전분포를 두고 random-walk Metropolis로 표본을 뽑아, 두 실험 설계를 비교한다.

```python {exec="browser" session="inverse"}
def rate(s_mm, vmax, km):                                    # Michaelis–Menten 반응 속도
    return vmax * s_mm / (km + s_mm)

def mcmc(s_mm, v_obs, noise=0.005, n_chain=16, n_step=3000, seed=0):   # 로그 파라미터의 random-walk Metropolis
    r = np.random.default_rng(seed)
    def logpost(th):                                         # th: (C, 2) = (log vmax, log km), 사전 N(0, 3^2)
        pred = rate(s_mm[None, :], np.exp(th[:, :1]), np.exp(th[:, 1:]))
        return -0.5 * np.sum((v_obs - pred)**2, axis=1) / noise**2 - 0.5 * np.sum(th**2, axis=1) / 3**2
    th = r.normal(0, 1, size=(n_chain, 2)); lp = logpost(th)
    out = []
    for i in range(n_step):
        prop = th + 0.1 * r.normal(size=th.shape)
        lp_prop = logpost(prop)
        acc = np.log(r.uniform(size=n_chain)) < lp_prop - lp
        th[acc], lp[acc] = prop[acc], lp_prop[acc]
        if i >= 1000:                                        # 앞 1000 step은 버린다
            out.append(th.copy())
    return np.exp(np.concatenate(out))                       # (n_chain * 2000, 2) = (vmax, km)

rng = np.random.default_rng(1)
q = lambda a, d: "–".join(f"{v:.{d}f}" for v in np.percentile(a, [5, 95]))   # 90% 구간 문자열
for name, s_mm in [("저농도만 0.1–0.5 mM", np.linspace(0.1, 0.5, 10)), ("넓게 0.1–20 mM    ", np.geomspace(0.1, 20, 10))]:
    v_obs = rate(s_mm, 2.0, 5.0) + 0.005 * rng.normal(size=10)   # 참값 vmax = 2, km = 5 mM
    post = mcmc(s_mm, v_obs)
    print(f"{name}: vmax {q(post[:, 0], 2)}  km {q(post[:, 1], 1)}  vmax/km {q(post[:, 0] / post[:, 1], 3)}"
          f"  v(20 mM) {q(rate(20.0, post[:, 0], post[:, 1]), 2)}  log 상관 {np.corrcoef(np.log(post.T))[0, 1]:.2f}")
```

```output
저농도만 0.1–0.5 mM: vmax 0.86–4.42  km 1.9–11.4  vmax/km 0.384–0.450  v(20 mM) 0.78–2.81  log 상관 1.00
넓게 0.1–20 mM    : vmax 1.98–2.01  km 4.9–5.1  vmax/km 0.396–0.405  v(20 mM) 1.59–1.60  log 상관 0.89
```

저농도만 잰 실험에서 $v_{\max}$와 $K_m$의 90% 구간은 상한이 하한의 5–6배일 만큼 넓고, 두 로그 파라미터의 상관이 1.00이다. 사후분포가 $v_{\max}/K_m = $ 일정한 선을 따라 길게 늘어선 능선이다. 그래도 비율 $v_{\max}/K_m$은 0.384–0.450으로 좁게 정해진다. 데이터가 정한 것은 비율이고, 개별 값의 범위는 사실상 사전분포가 정했다. 넓은 농도 범위를 재면 개별 파라미터도 1–2% 안으로 정해진다.

::quiz{id="nonidentifiable-posterior"}

이 상황에서 MAP 추정값 하나를 보고하면 능선 위의 임의의 한 점을 고른 셈이다. 사후분포 전체(또는 식별되는 조합과 그 구간)를 보고하고, 개별 파라미터가 사전분포에서 얼마나 움직였는지를 함께 보인다. 능선 모양의 사후분포는 random-walk MCMC가 섞이기 어려운 모양이기도 하다. 13.5장의 수렴 진단($\hat{R}$, 유효 표본 수)을 꼭 확인하고, 필요하면 식별 가능한 조합으로 재매개화한다.

**시뮬레이터 기반 추론.** 시뮬레이터가 확률적이거나(입자 물리의 검출기 시뮬레이션, 개체군 모델) 너무 복잡하면, 출력의 확률밀도 $p(\mathbf{y} \mid \boldsymbol{\theta})$를 계산할 수 없고 표본만 뽑을 수 있다. **시뮬레이터 기반 추론**(simulation-based inference, SBI)은 $(\boldsymbol{\theta}, \mathbf{y})$ 쌍을 시뮬레이션으로 많이 만든 뒤, 그 쌍으로 사후분포나 우도(또는 우도비)를 신경망 밀도 추정기로 학습하거나, 관측과 비슷한 출력을 낸 파라미터만 남기는 근사 베이지안 계산(ABC)을 쓴다(Cranmer, Brehmer, Louppe, 2020). 25.1장에서 자세히 다룬다.

## 민감도 분석

**왜 필요한가.** 시뮬레이터의 입력이 수십 개일 때, 어느 입력의 불확실성이 출력의 불확실성을 만드는지 알아야 측정 예산을 어디에 쓸지, 어느 파라미터를 고정해도 될지 정할 수 있다.

**국소 민감도.** 기준점 $\mathbf{x}_0$에서의 편미분 $\partial f/\partial x_i$(또는 단위를 없앤 $\partial \log f/\partial \log x_i$)다. 계산이 싸고 24.6장의 민감도 행렬과 같지만, **기준점 근처에서만** 유효하다. 입력의 불확실성 범위가 넓거나 모델이 비선형이면, 한 점의 기울기는 입력 범위 전체에서의 영향을 대표하지 못한다.

**전역 민감도: Sobol 지수.** 입력 $X_1, \dots, X_d$가 서로 독립인 확률변수이면 출력 $Y = f(\mathbf{X})$의 분산을 입력별·상호작용별 기여로 나눌 수 있다(Sobol, 2001). 두 지수가 핵심이다.

$$
S_i = \frac{\operatorname{Var}\big(\mathbb{E}[Y \mid X_i]\big)}{\operatorname{Var}(Y)}, \qquad
S_{T,i} = \frac{\mathbb{E}\big[\operatorname{Var}(Y \mid \mathbf{X}_{\sim i})\big]}{\operatorname{Var}(Y)}.
$$

$\mathbf{X}_{\sim i}$는 $X_i$를 뺀 나머지 입력이다. **1차 지수** $S_i$는 $X_i$ 하나만 정확히 알면 출력 분산이 평균적으로 얼마나 줄어드는지, **총 지수** $S_{T,i}$는 $X_i$를 뺀 나머지를 모두 알아도 남는 분산, 즉 $X_i$가 단독으로나 다른 입력과의 상호작용으로 만드는 분산 전체의 비율이다. $S_{T,i} - S_i$가 크면 상호작용이 크다. $S_{T,i} \approx 0$인 입력은 고정해도 된다.

**추정.** 두 독립 표본 행렬 $\mathbf{A}$, $\mathbf{B}$(각 $N \times d$)와, $\mathbf{A}$의 $i$번째 열만 $\mathbf{B}$의 것으로 바꾼 $\mathbf{A}_\mathbf{B}^{(i)}$로 Monte Carlo 추정한다. 비용은 모델 평가 $N(d + 2)$번이다. 아래 코드는 Saltelli 외(2010)가 비교한 추정식 중 1차 지수 $\frac{1}{N}\sum f(\mathbf{B})\big(f(\mathbf{A}_\mathbf{B}^{(i)}) - f(\mathbf{A})\big)/V$와 총 지수 $\frac{1}{2N}\sum \big(f(\mathbf{A}) - f(\mathbf{A}_\mathbf{B}^{(i)})\big)^2/V$를 쓴다.

**코드.** 민감도 분석의 표준 시험 함수인 Ishigami 함수 $f = \sin x_1 + 7\sin^2 x_2 + 0.1\,x_3^4 \sin x_1$($x_i \sim U(-\pi, \pi)$)는 지수의 해석해가 알려져 있다($S_1 \approx 0.314$, $S_2 \approx 0.442$, $S_3 = 0$, $S_{T,1} \approx 0.558$, $S_{T,2} \approx 0.442$, $S_{T,3} \approx 0.244$). 원점의 국소 민감도와 비교한다.

```python {exec="browser" session="inverse"}
def ishigami(X, a=7.0, b=0.1):                               # X: (N, 3), 각 성분 U(-pi, pi)
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1])**2 + b * X[:, 2]**4 * np.sin(X[:, 0])

x0, h = np.zeros((1, 3)), 1e-5
grad = [(ishigami(x0 + h * np.eye(3)[[i]]) - ishigami(x0 - h * np.eye(3)[[i]]))[0] / (2 * h) for i in range(3)]
print("원점에서의 국소 민감도 df/dx_i:", ", ".join(f"{g:.3f}" for g in grad))
rng = np.random.default_rng(0)
N = 100_000
A_s, B_s = rng.uniform(-np.pi, np.pi, (N, 3)), rng.uniform(-np.pi, np.pi, (N, 3))
fA, fB = ishigami(A_s), ishigami(B_s)
V = np.var(np.concatenate([fA, fB]))                         # 출력의 총분산
for i in range(3):
    ABi = A_s.copy(); ABi[:, i] = B_s[:, i]                  # A에서 i번째 열만 B로 바꾼 표본
    fABi = ishigami(ABi)
    S1 = max(np.mean(fB * (fABi - fA)) / V, 0.0)             # 1차 지수, 추정 잡음으로 생긴 음수는 0으로
    ST = 0.5 * np.mean((fA - fABi)**2) / V                   # 총 지수
    print(f"x{i + 1}: S1 = {S1:.2f}   ST = {ST:.2f}")
```

```output
원점에서의 국소 민감도 df/dx_i: 1.000, 0.000, 0.000
x1: S1 = 0.31   ST = 0.55
x2: S1 = 0.44   ST = 0.44
x3: S1 = 0.00   ST = 0.24
```

원점의 국소 민감도는 $x_1$만 중요하고 $x_2$, $x_3$는 영향이 없다고 말한다. 전역 지수는 정반대에 가깝다. 출력 분산의 44%를 $x_2$가 만들고, $x_3$는 혼자서는 기여가 없지만($S_3 = 0$) $x_1$과의 상호작용으로 24%를 만든다. 원점에서 $\sin^2 x_2$의 기울기와 $x_3^4$의 기울기가 0일 뿐이다. 추정값은 $N = 10^5$에서 해석해와 소수 둘째 자리까지 맞는다.

::quiz{id="sobol-interaction"}

**연구에서의 의미.** Sobol 지수는 입력의 분포(범위)를 정해야 계산되고, 그 분포에 따라 결과가 바뀐다. 입력끼리 상관이 있으면 위의 분해가 그대로 성립하지 않으므로 다른 방법이 필요하다. 모델 평가가 비싸면 $N(d+2)$번의 실행이 부담이므로, 대리모델(24.5장)을 먼저 만들고 그 위에서 지수를 계산한다. 이때는 대리모델의 오차가 지수에 섞인다는 점을 함께 보고한다.

## 불확실성 전파

**왜 필요한가.** 파라미터를 추정한 목적은 대개 예측이다. 파라미터가 불확실하면 예측도 불확실하고, 그 크기는 예측하려는 조건에 따라 크게 다르다.

**정의.** 사후분포 $p(\boldsymbol{\theta} \mid \mathbf{y})$를 예측 $y_*$로 전파한 **사후 예측분포**는

$$
p(y_* \mid \mathbf{y}) = \int p(y_* \mid \boldsymbol{\theta})\,p(\boldsymbol{\theta} \mid \mathbf{y})\,d\boldsymbol{\theta}
$$

다. **Monte Carlo 전파**는 사후 표본 $\boldsymbol{\theta}^{(s)}$마다 모델을 실행하고 측정 잡음을 더해 $y_*^{(s)}$를 뽑은 뒤, 그 분위수로 예측 구간을 만든다. 선형화(delta method) $\operatorname{Var}(y_*) \approx \mathbf{J}\boldsymbol{\Sigma}_\theta\mathbf{J}^\top$는 싸지만, 파라미터 불확실성이 넓고 모델이 비선형이면 틀린다. 비선형 모델에서는 평균 파라미터로 계산한 예측 $f(\mathbb{E}[\boldsymbol{\theta}])$이 예측의 평균 $\mathbb{E}[f(\boldsymbol{\theta})]$과 다르다는 점(Jensen 부등식)도 기억한다.

**코드로 본 의미.** cell 4의 마지막 열이 Monte Carlo 전파다. 저농도만 잰 실험의 사후 표본을 측정 범위 밖 $S = 20$ mM로 밀면 예측 속도의 90% 구간이 0.78–2.81로, 참값 1.6을 포함하지만 상한이 하한의 3.6배다. 측정 범위 안에서는 모든 사후 표본이 데이터를 똑같이 잘 설명했는데, 범위 밖에서 서로 크게 갈라진다. 넓은 범위를 잰 실험에서는 같은 예측이 1.59–1.60으로 좁다. **식별되지 않은 방향은 측정 범위 안에서는 보이지 않다가, 외삽에서 예측의 불확실성으로 나타난다.**

불확실성의 출처는 셋으로 나눠 보고한다. 파라미터 불확실성(위의 전파), 측정 잡음(aleatoric, 12.3장), 모델 자체의 오차(24.5장의 불일치 함수 $\delta$)다. 마지막 것은 사후분포에 자동으로 들어가지 않는다. 모델이 틀렸다면 파라미터의 사후분포가 아무리 좁아도 예측은 틀린다.

## 실습

시뮬레이터 파라미터를 측정 데이터로 베이지안 추정하고, 사후 불확실성을 예측으로 전파한다. 코드는 `labs/scientific-ml/inverse-problems-and-uq/posterior_predictive.py`다. cell 4의 Michaelis–Menten 실험을 다른 잡음으로 40번 반복하고, 측정 범위 밖 $S = 20$ mM에서 새로 측정할 속도의 90% 구간이 실제로 그 값을 포함하는 비율을 잰다. 두 보고 방식을 비교한다.

- **점추정 + 잡음**: 사후 표본 중 로그 사후밀도가 가장 높은 파라미터로 예측하고, 구간은 측정 잡음만으로 만든다.
- **사후 예측**: 사후 표본마다 예측하고 잡음을 더한 표본의 분위수로 구간을 만든다.

집필 시점의 실행 결과는 다음과 같다.

```output
S = 20.0 mM 예측의 90% 구간 포함률 (실험 40번 반복, 명목 0.90)
  저농도만 0.1–0.5 mM   점추정+잡음 0.03   사후 예측 0.97 (평균 폭 2.342)
  넓게 0.1–20 mM      점추정+잡음 0.82   사후 예측 0.93 (평균 폭 0.021)
```

- **식별이 안 되는 실험에서 점추정 구간은 거의 쓸모가 없다.** 명목 90%인 구간이 실제로는 3%만 맞았다. 측정 범위 안에서 데이터를 잘 맞춘 파라미터가 범위 밖에서는 임의의 값을 낸다.
- **사후 예측은 포함률을 지키지만 정직하게 넓다.** 평균 폭 2.342는 "이 실험으로는 20 mM의 속도를 알 수 없다"는 정확한 보고다. 이 폭을 줄이는 방법은 계산이 아니라 측정 범위를 넓히는 실험이다.
- **식별되는 실험에서도 점추정 구간은 약간 좁다**(0.82). 파라미터 추정 오차를 무시했기 때문이다.

과제는 자신이 다루는 시뮬레이터(또는 이 예제)에서 같은 비교를 하는 것이다. 사전분포를 바꿔(예: 표준편차 3 → 1) 사후 예측 구간이 얼마나 바뀌는지 보고, 그 변화가 크다면 "결론이 사전분포에 의존한다"는 사실을 결과와 함께 쓴다. 이어서 연습문제로 SVD 필터 인자를 이용한 Tikhonov 해를 구현한다.

::exercise{id="tikhonov-svd"}

## 실패 사례

- **점추정 파라미터로 예측하고 불확실성을 보고하지 않는 것.** 실습에서 식별이 안 되는 실험의 점추정 구간은 명목 90%에서 실제 3%만 맞았다. 역문제의 결과는 사후분포(또는 최소한 식별되는 조합의 구간)와 예측 구간으로 보고한다.
- **참값을 보고 정규화 강도를 고르는 것.** 모의 실험에서 오차가 최소인 $\lambda$를 골라 보고하면, 참값을 모르는 실제 문제에서 재현할 수 없는 성능을 주장하는 셈이다. L-curve·불일치 원리·GCV처럼 데이터만 쓰는 기준으로 고르고, 모의 실험에서는 그 기준이 최적에서 얼마나 떨어지는지를 보고한다.
- **점마다의 구간 띠를 해 전체의 불확실성으로 읽는 것.** 점마다의 95% 구간을 이어 그린 띠 안에 있는 곡선이 모두 그럴듯한 해인 것은 아니다. 연구 질문이 묻는 양(평균, 적분, 위치)의 사후분포를 따로 계산한다.
- **국소 민감도로 입력을 고정하는 것.** 기준점에서 기울기가 0인 입력도 전역적으로는 출력 분산의 대부분을 만들 수 있다(Ishigami의 $x_2$, $x_3$).

## 통과 기준

- 역문제의 결과를 사후분포와 예측 구간으로 보고하고, 모의 데이터에서 예측 구간의 포함률을 확인한다.
- 정규화 강도를 데이터만 쓰는 기준(L-curve, 불일치 원리, GCV 중 둘 이상)으로 고르고, 고른 값이 서로 일치하는지 보고한다.
- 입력이 셋 이상인 모델에서 Sobol 1차·총 지수를 추정하고, 국소 민감도와 다른 결론이 나오는지 확인한다.
- 연습문제 `tikhonov-svd`의 테스트를 통과한다.

## 참고 문헌

- Cranmer, Brehmer, Louppe (2020). The frontier of simulation-based inference. *PNAS*, 117(48), 30055–30062. — 우도를 계산할 수 없는 시뮬레이터의 추론 방법 개관
- Hansen, O'Leary (1993). The use of the L-curve in the regularization of discrete ill-posed problems. *SIAM Journal on Scientific Computing*, 14(6), 1487–1503. — L-curve와 모서리로 정규화 강도를 고르는 방법
- Saltelli, Annoni, Azzini, Campolongo, Ratto, Tarantola (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications*, 181(2), 259–270. — Sobol 지수의 Monte Carlo 추정 설계와 추정식 비교
- Sobol' (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1–3), 271–280. — 분산 분해에 기반한 전역 민감도 지수
- Stuart (2010). Inverse problems: A Bayesian perspective. *Acta Numerica*, 19, 451–559. — 역문제의 베이지안 정식화
