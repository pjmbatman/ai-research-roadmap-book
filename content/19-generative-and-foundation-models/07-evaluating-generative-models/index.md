---
id: generative-and-foundation-models/evaluating-generative-models
title: "생성모델의 평가"
title_en: "Evaluating Generative Models"
tier: core
status: draft
prerequisites: [generative-and-foundation-models/generative-modeling]
---

# 생성모델의 평가

분류 모델은 시험 자료의 정답과 비교하면 평가가 끝난다. 생성모델에는 그런 정답이 없다. 모델이 내놓은 표본 하나하나는 "맞다", "틀리다"로 채점할 수 없고, 평가해야 할 것은 표본들이 이루는 **분포**가 데이터의 분포를 얼마나 잘 닮았는가다. 그런데 분포의 닮음은 한 숫자로 요약되지 않는다. 표본 하나하나가 그럴듯한가, 데이터의 모든 영역을 덮는가, 학습 자료를 베낀 것은 아닌가는 서로 다른 질문이고, 한 질문에 좋은 점수를 받는 모델이 다른 질문에서 실패하는 일은 흔하다.

이 장은 이 질문들을 하나씩 떼어 재는 지표를 만들고, 각 지표가 **속는** 상황을 직접 만들어 본다. 그리고 과학 데이터에서는 일반 지표만으로 부족하다는 것, 즉 물리 법칙과 통계적 구조를 따로 점검하고, 생성 데이터가 실제 연구 과제에 쓸모 있는지를 하위 과제로 검증해야 한다는 것을 보인다. 19.1장에서 "보기 좋은 표본은 분포 학습의 증거가 아니다"라고 했던 경고를 구체적인 측정으로 바꾸는 장이다.

## 학습 목표

- 생성 결과의 품질·다양성(커버리지)·기억을 구분해 각각 측정한다
- likelihood·MMD·FID 같은 분포 지표가 무엇을 보고 무엇을 놓치는지 설명하고, 지표가 속는 사례를 찾아낸다
- 도메인 지표와 하위 과제로 생성 데이터의 유용성을 검증한다

## 평가 축

**왜 필요한가.** 생성모델이 실패하는 방식은 여러 가지다. 표본이 흐릿하거나 물리적으로 불가능할 수 있고(품질), 데이터의 일부 모드만 만들 수 있고(다양성·커버리지), 학습 자료를 거의 그대로 내놓을 수도 있다(기억). 하나의 지표로 이 실패들을 모두 잡을 수는 없다. 그래서 평가는 축을 나누는 데서 시작한다.

| 축 | 질문 | 실패의 예 |
|---|---|---|
| 품질(fidelity, precision) | 생성된 표본이 실제 데이터가 있을 법한 곳에 있는가 | 흐릿한 이미지, 결합 길이가 불가능한 분자 |
| 다양성·커버리지(recall) | 실제 데이터의 모든 영역을 생성하는가 | mode collapse, 드문 극값 사건을 만들지 못함 |
| 기억(memorization) | 학습 자료를 베끼지 않고 새 표본을 만드는가 | 학습 표본의 복사본, 개인 정보 노출 |
| 유용성 | 생성 데이터가 실제 과제에 도움이 되는가 | 보기에는 좋지만 하위 모델 학습에 쓸모없음 |

**정의: precision과 recall.** 실제 표본과 생성 표본을 같은 특징 공간에 놓고, 각 표본의 $k$번째 최근접 이웃까지의 거리를 반지름으로 하는 공들의 합집합으로 각 분포의 지지집합(다양체)을 근사한다. 생성 표본 중 실제 다양체 안에 들어가는 비율이 **precision**, 실제 표본 중 생성 다양체 안에 들어가는 비율이 **recall**이다(Kynkäänniemi 외, 2019). FID 같은 단일 수치가 품질과 다양성을 한 숫자로 섞어 버리는 문제를 풀기 위해 제안되었다.

**정의: 기억의 탐지.** 생성 표본에서 가장 가까운 **학습** 표본까지의 거리를, 학습에 쓰지 않은 실제 표본에서 가장 가까운 학습 표본까지의 거리와 비교한다. 새 표본을 만드는 모델이라면 두 거리의 분포가 비슷해야 하고, 학습 자료를 베끼는 모델이라면 생성 표본이 학습 표본에 비정상적으로 가깝다.

**코드.** 2차원 혼합 가우시안(모드 4개)을 데이터로 두고, 네 가지 "생성모델"을 흉내 낸다. 데이터 분포에서 새로 뽑은 좋은 모델, 모드 하나를 빠뜨린 모델, 각 모드를 두 배로 퍼뜨린 흐릿한 모델, 학습 자료에 아주 작은 잡음만 더해 내놓는 복사 모델이다.

```python {exec="browser" session="ev"}
import numpy as np

rng = np.random.default_rng(0)
modes = np.array([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])   # 2차원 혼합 가우시안의 중심 (4, 2)
def real(n, std=0.4, use=4):
    return modes[rng.integers(use, size=n)] + std * rng.normal(size=(n, 2))

train, test = real(500), real(500)                 # 생성모델의 학습 자료, 평가용 실제 자료 (500, 2)
models = {"좋은 모델": real(500), "모드 하나 누락": real(500, use=3),
          "흐릿함(std 2배)": real(500, std=0.8), "학습 자료 복사": train + 0.02 * rng.normal(size=(500, 2))}

def dists(a, b):                                   # 쌍별 유클리드 거리 (len(a), len(b))
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))

def precision_recall(real_x, fake_x, k=3):         # k-NN 공 합집합으로 만든 다양체 근사
    r_real = np.sort(dists(real_x, real_x), 1)[:, k]   # 각 실제 점의 k번째 이웃 거리 (자기 자신 제외)
    r_fake = np.sort(dists(fake_x, fake_x), 1)[:, k]
    prec = (dists(fake_x, real_x) <= r_real[None, :]).any(1).mean()   # 가짜가 실제 다양체 안에 있는가
    rec = (dists(real_x, fake_x) <= r_fake[None, :]).any(1).mean()    # 실제가 가짜 다양체 안에 있는가
    return prec, rec

base_nn = np.median(dists(test, train).min(1))     # 실제 새 표본이 학습 자료에서 떨어진 거리
print("모델               precision  recall   학습 자료까지 거리 / 기준")
for name, fake in models.items():
    p, r = precision_recall(test, fake)
    print(f"{name:14}     {p:.3f}     {r:.3f}    {np.median(dists(fake, train).min(1)) / base_nn:.2f}")
```

```output
모델               precision  recall   학습 자료까지 거리 / 기준
좋은 모델              0.972     0.960    0.92
모드 하나 누락           0.986     0.680    0.98
흐릿함(std 2배)        0.698     0.992    2.51
학습 자료 복사           0.970     0.970    0.34
```

세 지표가 각자 다른 실패를 잡는다.

- **모드 하나 누락**은 precision이 높고(0.986) recall만 낮다(0.680). 만든 것은 모두 그럴듯하지만, 실제 데이터의 4분의 1을 만들지 못한다. 표본을 몇 개 골라 보는 방식으로는 절대 찾을 수 없는 실패다.
- **흐릿함**은 반대로 recall은 높고(0.992) precision이 낮다(0.698). 넓게 퍼뜨린 덕에 모든 모드를 덮지만, 표본의 30%가 데이터가 없는 곳에 떨어진다.
- **학습 자료 복사**는 precision과 recall이 좋은 모델과 똑같다(0.970, 0.970). 분포의 관점에서는 학습 자료 자체가 훌륭한 표본이기 때문이다. 이 모델을 잡는 것은 학습 자료까지의 거리뿐이다. 좋은 모델의 비율은 0.92로 실제 새 표본과 비슷하지만, 복사 모델은 0.34로 학습 표본에 세 배 가깝게 붙어 있다.

::quiz{id="precision-recall-failure"}

**연구에서의 의미.** 기억은 단순한 성능 문제가 아니다. Carlini 외(2023)는 공개된 이미지 확산 모델에서 생성–필터링 절차로 학습 이미지 천 개 이상을 거의 그대로 뽑아냈고, 학습 자료의 중복 제거나 차분 프라이버시 학습 같은 기존 대응이 만족스러운 프라이버시–유용성 균형을 주지 못한다고 보고했다. 환자 기록, 미공개 실험 자료처럼 민감한 자료로 생성모델을 학습할 때는 기억 점검이 평가의 필수 항목이다. 다만 위의 거리 비율은 특징 공간과 거리의 선택에 의존한다. 이미지 픽셀 거리로는 잘라 붙이거나 조금 변형한 복사본을 놓칠 수 있다.

:::note[분야 예시]
신약 후보 분자를 생성하는 연구에서는 품질(화학적으로 타당한 구조의 비율), 다양성(생성된 분자들 사이의 구조적 차이), 새로움(학습 데이터베이스에 없는 분자의 비율)을 따로 보고하는 것이 관행이다. 세 번째 항목이 기억 점검에 해당한다. 학습 데이터베이스의 분자를 그대로 내놓는 모델은 타당성과 물성 분포가 완벽해 보여도 새 후보를 하나도 제안하지 못한다.
:::

## 지표

**likelihood.** 모델이 밀도를 계산할 수 있으면(자기회귀, flow, VAE의 하한), held-out 자료의 평균 log-likelihood가 가장 원칙적인 지표다. 14.1장에서 본 것처럼 그 음수는 데이터 분포와의 cross-entropy이고, 이를 최소화하는 것은 forward KL을 줄이는 것이다. 그런데 likelihood와 표본 품질은 고차원에서 거의 독립적으로 움직일 수 있다(Theis 외, 2016). 다음 논증이 핵심이다. 좋은 모델 $p$에 쓰레기 분포 $q_{\text{junk}}$를 섞어 $q = 0.01\,p + 0.99\,q_{\text{junk}}$를 만들면, 표본의 99%가 쓰레기인데도

$$
\log q(\mathbf{x}) \ \ge\ \log 0.01 + \log p(\mathbf{x}) = \log p(\mathbf{x}) - \log 100
$$

이다. log-likelihood는 표본당 최대 $\log 100 \approx 4.6$ nat만 줄어든다. 차원이 크면 차원당으로는 거의 차이가 없다.

```python {exec="browser" session="ev"}
from scipy.stats import multivariate_normal

d = 64
x_te = rng.normal(size=(2000, d))                  # 실제 자료: N(0, I_64)
logp_good = multivariate_normal(np.zeros(d), np.eye(d)).logpdf(x_te)
logp_junk = multivariate_normal(np.zeros(d), 25 * np.eye(d)).logpdf(x_te)
for w in [1.0, 0.5, 0.01]:                         # 좋은 모델을 w만큼, 쓰레기(분산 25배)를 1-w만큼 섞는다
    mix = np.logaddexp(np.log(w) + logp_good, np.log1p(-w) + logp_junk) if w < 1 else logp_good
    print(f"좋은 표본 비율 {w:4.2f}   평균 log-likelihood/차원 {mix.mean() / d:.4f} nat")
print(f"상한 차이 log(100)/d = {np.log(100) / d:.4f} nat")
```

```output
좋은 표본 비율 1.00   평균 log-likelihood/차원 -1.4201 nat
좋은 표본 비율 0.50   평균 log-likelihood/차원 -1.4309 nat
좋은 표본 비율 0.01   평균 log-likelihood/차원 -1.4920 nat
상한 차이 log(100)/d = 0.0720 nat
```

64차원에서 표본의 99%를 쓰레기로 바꾸어도 차원당 log-likelihood는 −1.4201에서 −1.4920으로, 상한 0.072 nat 안에서만 나빠진다. 이미지처럼 수만 차원이면 그 차이는 소수점 아래 넷째 자리에 묻힌다. 반대 방향도 성립한다. 아주 좋은 표본을 내는 모델이 소수의 test 점에 거의 0의 확률을 주면 log-likelihood가 크게 나빠질 수 있다. 그래서 **likelihood가 좋다는 것은 표본이 좋다는 증거가 아니고, 그 역도 아니다.** likelihood는 커버리지(데이터의 어느 곳에도 확률 0을 주지 않음)에 민감하고, 품질에는 둔감하다.

**MMD와 2-표본 검정.** 밀도를 계산할 수 없는 모델(GAN, 확산 모델의 표본)은 표본만으로 평가해야 한다. 가장 깔끔한 도구는 **2-표본 검정**(two-sample test)이다. "생성 표본과 실제 표본이 같은 분포에서 왔다"는 귀무가설을 검정한다. **maximum mean discrepancy**(MMD)는 kernel $k$로 두 분포의 차이를 잰다(Gretton 외, 2012).

$$
\mathrm{MMD}^2(P, Q) = \mathbb{E}[k(\mathbf{x}, \mathbf{x}')] + \mathbb{E}[k(\mathbf{y}, \mathbf{y}')] - 2\,\mathbb{E}[k(\mathbf{x}, \mathbf{y})], \qquad \mathbf{x}, \mathbf{x}' \sim P,\ \mathbf{y}, \mathbf{y}' \sim Q.
$$

RBF kernel $k(\mathbf{x}, \mathbf{y}) = \exp(-\lVert \mathbf{x} - \mathbf{y} \rVert^2 / 2h^2)$처럼 특성적인(characteristic) kernel이면 $\mathrm{MMD} = 0$은 $P = Q$와 동치다. 표본 추정에서 같은 점끼리의 항 $k(\mathbf{x}_i, \mathbf{x}_i)$를 빼면 불편 추정량이 되고, 그래서 두 분포가 같을 때 음수가 나올 수도 있다. 귀무분포는 **순열 검정**(permutation test)으로 얻는다. 귀무가설이 참이면 어느 표본이 생성 쪽이고 어느 표본이 실제 쪽인지는 의미가 없으므로, 두 표본을 합쳐 무작위로 다시 나눈 뒤 같은 통계량을 계산하기를 반복하면 귀무가설 아래의 분포가 나온다. $p$ 값은 원래 통계량 이상인 순열의 비율이다. 대역폭 $h$로는 흔히 표본 사이 거리의 중앙값을 쓴다(median heuristic). 모델이 각 모드를 1.5배 넓게 만든 경우를 검정해 본다. 각 설정에서 실험을 20번 반복해 $p < 0.05$로 기각한 비율을 센다.

```python {exec="browser" session="ev"}
def mmd2_with_perms(x, y, bw=None, n_perm=100):    # 불편 MMD²와 순열 null 분포, RBF kernel
    z, n = np.vstack([x, y]), len(x)
    D2 = dists(z, z) ** 2                          # (2n, 2n)
    bw = np.sqrt(np.median(D2[D2 > 0])) if bw is None else bw   # 없으면 거리의 중앙값
    K = np.exp(-D2 / (2 * bw**2))                  # 대각 = 1
    E = np.zeros((2 * n, n_perm + 1))              # 열마다 "X 쪽" 표시: 0열 = 원래 배정
    E[:n, 0] = 1
    for j in range(1, n_perm + 1):
        E[rng.permutation(2 * n)[:n], j] = 1
    F = 1 - E
    kxx, kyy, kxy = (E * (K @ E)).sum(0), (F * (K @ F)).sum(0), (E * (K @ F)).sum(0)
    m = (kxx - n + kyy - n) / (n * (n - 1)) - 2 * kxy / n**2    # 대각(자기 자신) 항을 뺀 불편 추정
    return m[0], m[1:]

def reject_rate(std, bw, n, reps=20):              # 20번 반복한 실험 중 p < 0.05인 비율
    out = []
    for _ in range(reps):
        stat, null = mmd2_with_perms(real(n, std=std), real(n), bw)
        out.append((1 + np.sum(null >= stat)) / (1 + len(null)) < 0.05)
    return np.mean(out)

print("대역폭      n    기각률(같은 분포)   기각률(std 1.5배)")
for bw in [None, 0.5]:
    for n in [30, 200]:
        print(f"{'중앙값' if bw is None else bw:>5}    {n:4d}        {reject_rate(0.4, bw, n):.2f}              {reject_rate(0.6, bw, n):.2f}")
```

```output
대역폭      n    기각률(같은 분포)   기각률(std 1.5배)
  중앙값      30        0.05              0.15
  중앙값     200        0.00              0.05
  0.5      30        0.00              0.10
  0.5     200        0.00              1.00
```

- **같은 분포에서는 어느 설정이든 기각률이 0.05 이하다.** 순열 검정이 1종 오류를 지킨다는 확인이다.
- **중앙값 대역폭은 표본이 200개여도 차이를 거의 못 찾는다(0.05).** 모드 사이 거리(4)가 지배하는 중앙값 대역폭은 모드 안의 퍼짐(0.4 대 0.6)을 보기에는 너무 넓다. kernel이 모드 하나를 점 하나처럼 본다.
- **대역폭을 차이의 척도(0.5)에 맞추면 200개로 모든 실험에서 기각한다(1.00).** 그러나 표본이 30개면 같은 대역폭으로도 10%만 기각한다.

"MMD 검정에서 유의한 차이가 없었다"는 결과는 대역폭과 표본 수가 그 차이를 볼 수 있을 때만 의미가 있다. 검정력이 낮은 설정에서 기각하지 못한 것을 "모델이 데이터 분포를 재현했다"는 증거로 쓰면 안 된다. 대역폭을 여러 개 쓰거나, 자료의 일부로 대역폭을 고르고 나머지로 검정하는 것이 대응이다(대역폭을 고른 자료로 검정까지 하면 1종 오류가 깨진다).

**FID.** 이미지 생성 연구에서 가장 널리 쓰인 지표는 **Fréchet Inception Distance**(FID)다(Heusel 외, 2017). 실제 이미지와 생성 이미지를 사전학습된 이미지 분류망(Inception)에 넣어 중간 특징을 얻고, 두 특징 집합에 각각 가우시안을 맞춘 뒤 두 가우시안 사이의 Fréchet 거리(가우시안 사이의 2-Wasserstein 거리의 제곱)를 계산한다.

$$
\mathrm{FID} = \lVert \boldsymbol{\mu}_r - \boldsymbol{\mu}_g \rVert^2 + \operatorname{tr}\big(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r\boldsymbol{\Sigma}_g)^{1/2}\big).
$$

$\boldsymbol{\mu}, \boldsymbol{\Sigma}$는 특징의 평균과 공분산이다. 두 가지가 결과를 정한다. 첫째, **특징**이 무엇을 보는가. 둘째, 특징 분포를 **가우시안으로 요약**해도 되는가. 같은 2차원 자료에서, 특징으로 원 좌표를 그대로 쓴 경우와 작은 무작위 ReLU 특징 32개를 쓴 경우를 비교한다. 위의 네 모델에 "평균과 공분산만 학습 자료에 맞춘 가우시안"을 더한다. 이 모델은 모드가 하나도 없는 덩어리다.

```python {exec="browser" session="ev"}
from scipy.linalg import sqrtm

def frechet(a, b):                                 # 두 특징 집합에 가우시안을 맞춘 뒤의 Fréchet 거리
    mu_a, mu_b = a.mean(0), b.mean(0)
    Sa, Sb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = sqrtm(Sa @ Sb).real                  # (Sa Sb)^{1/2}
    return np.sum((mu_a - mu_b) ** 2) + np.trace(Sa + Sb - 2 * covmean)

W_f = rng.normal(size=(2, 32))
feat = lambda x: np.maximum(0, x @ W_f + 0.5)      # 작은 무작위 ReLU 특징 (N, 32)
gens = {"좋은 모델": lambda n: real(n), "모드 하나 누락": lambda n: real(n, use=3),
        "흐릿함(std 2배)": lambda n: real(n, std=0.8),
        "학습 자료 복사": lambda n: train[:n] + 0.02 * rng.normal(size=(n, 2)),
        "평균·공분산만 맞춘 가우시안": lambda n: train.mean(0) + rng.normal(size=(n, 2)) @ np.linalg.cholesky(np.cov(train.T)).T}
print("모델 (5회 반복 평균)          원 좌표   ReLU 특징   ReLU 특징·학습 자료 기준")
for name, g in gens.items():
    res = np.array([(frechet(f, r), frechet(feat(f), feat(r)), frechet(feat(f), feat(train)))
                    for f, r in ((g(500), real(500)) for _ in range(5))]).mean(0)
    print(f"{name:22}   {res[0]:6.3f}    {res[1]:6.2f}       {res[2]:6.2f}")
for n in (50, 500):
    same = [frechet(feat(real(n)), feat(real(n))) for _ in range(5)]
    print(f"같은 분포 두 표본 n = {n:3d}: FID(ReLU 특징) {np.mean(same):.2f}")
```

```output
모델 (5회 반복 평균)          원 좌표   ReLU 특징   ReLU 특징·학습 자료 기준
좋은 모델                     0.042      0.43         0.47
모드 하나 누락                  1.495     23.55        27.52
흐릿함(std 2배)               0.061      1.77         2.18
학습 자료 복사                  0.101      0.95         0.00
평균·공분산만 맞춘 가우시안           0.117      8.58         8.19
같은 분포 두 표본 n =  50: FID(ReLU 특징) 6.14
같은 분포 두 표본 n = 500: FID(ReLU 특징) 0.41
```

이 표에서 FID가 속는 방식이 네 가지 보인다.

- **특징이 약하면 FID가 거의 아무것도 구별하지 못한다.** 원 좌표에서 흐릿한 모델(0.061)과 모드가 없는 가우시안(0.117)은 좋은 모델(0.042)과 큰 차이가 없다. 원 좌표의 평균·공분산은 모드 사이의 큰 퍼짐이 지배하고, 가우시안 모델은 바로 그 평균·공분산을 맞추도록 만들어졌다. 특징이 분포의 구조를 담아야 FID가 의미를 가진다. 이미지에서 Inception 특징을 쓰는 이유이고, 동시에 과학 데이터에 이미지 분류망의 특징을 그대로 쓰면 안 되는 이유다.
- **좋은 특징에서도 가우시안 요약은 정보를 버린다.** ReLU 특징에서는 가우시안 모델이 8.58로 드러나지만, 이것은 특징의 비선형성 덕분이다. 특징 분포 자체의 다봉 구조는 평균과 공분산 두 가지로 요약된다.
- **학습 자료와 비교하면 복사 모델이 최고점을 받는다.** 학습 자료 기준 FID는 복사 모델이 0.00으로 가장 좋다. held-out 실제 자료와 비교해야 하고, 그래도 복사 모델(0.95)은 좋은 모델(0.43)과 같은 자릿수다. FID에는 베끼기를 벌하는 항이 없다.
- **표본 수에 따라 값이 크게 바뀐다.** 같은 분포의 두 표본도 50개씩이면 6.14, 500개씩이면 0.41이다. 표본 공분산의 추정 오차가 체계적인 양의 편향을 만들기 때문이다. 표본 수가 다른 FID끼리는 비교할 수 없다.

::quiz{id="fid-memorization"}

**Wasserstein 거리.** 14.1장의 $W_1$은 지지집합이 겹치지 않아도 의미 있는 값을 주므로 생성모델의 분포 거리로도 쓰인다. 1차원(주변분포 하나)에서는 정렬로 정확히 계산되고, 19.6장에서 이 방법으로 표본 품질을 쟀다. 고차원에서는 표본 추정이 어렵고 표본 수에 대한 수렴이 느리므로, 무작위 1차원 투영의 $W_1$을 평균하는 sliced 버전이 흔히 쓰인다.

| 지표 | 필요한 것 | 잘 잡는 것 | 놓치는 것 |
|---|---|---|---|
| held-out log-likelihood | 밀도 계산 | 커버리지(확률 0인 영역) | 표본 품질, 기억 |
| precision / recall | 특징, 최근접 이웃 | 품질과 커버리지를 따로 | 기억, 특징 밖의 구조 |
| 학습 자료까지 거리 | 특징, 학습 자료 | 기억 | 품질, 커버리지 |
| MMD 검정 | kernel, 대역폭 | 분포 차이의 유무(검정력 안에서) | 차이의 종류, 대역폭 밖의 척도 |
| FID | 특징, 많은 표본 | 특징 평균·공분산의 차이 | 가우시안 밖의 구조, 기억 |

::widget{name="gen-metric-failures"}

## 도메인 지표

**왜 필요한가.** 위의 지표들은 모두 "특징 공간에서의 거리"라는 일반적인 틀에 기대고, 과학 데이터에서 중요한 성질을 모른다. 기후장은 질량과 에너지를 보존해야 하고, 분자는 결합 규칙을 지켜야 하며, 시계열은 시간 상관과 스펙트럼을 가져야 한다. 일반 지표가 좋은데 이런 성질이 깨진 표본은 연구에 쓸 수 없다.

**두 종류의 점검.**

- **물리 법칙의 만족**: 보존량(질량·에너지·전하)의 잔차, 경계 조건, 단위와 값의 범위, 대칭성. 9.7장의 시뮬레이터 검증(V&V)과 같은 질문을 생성 표본에 던지는 것이다. 표본마다 위반량을 계산해 그 분포를 보고한다.
- **통계적 특성의 일치**: 시점별 평균·분산 같은 1차 통계뿐 아니라 자기상관, 스펙트럼, 공간 상관, 극값 분포처럼 **여러 값 사이의 관계**를 본다. 1차 통계만 맞추는 것은 쉽고, 관계는 쉽게 깨진다.

**코드.** 감쇠 진동 시계열(18.8장과 같은 형태, 진동수 3–8 Hz)을 실제 자료로 두고, 세 "생성모델"을 비교한다. 좋은 모델, 시점마다 다른 실제 계열에서 값을 가져와 이어 붙인 모델(시점별 분포는 완벽하다), 물리는 맞지만 진동수를 3–5 Hz에서만 만드는 모델이다. 마지막 열은 다음 절의 하위 과제 평가다.

```python {exec="browser" session="ev"}
t_s = np.linspace(0, 1, 64)                        # 시간 (T,), 단위 s
def oscillators(n, f_hz=(3, 8)):                   # 감쇠 진동 (n, T)과 각진동수 ω (n,), 단위 rad/s
    om = 2 * np.pi * rng.uniform(*f_hz, n); ga = rng.uniform(0.5, 4.0, n); ph = rng.uniform(0, 2 * np.pi, n)
    return np.exp(-ga[:, None] * t_s) * np.cos(om[:, None] * t_s + ph[:, None]) + 0.05 * rng.normal(size=(n, 64)), om

X_real, om_real = oscillators(1000); X_te, om_te = oscillators(1000)
X_shuf = np.stack([col[rng.permutation(1000)] for col in X_real.T], 1)   # 시점마다 다른 계열에서 값을 가져온다
gens_ts = {"좋은 모델": oscillators(1000), "시점별 섞기": (X_shuf, om_real), "좁은 주파수(3–5 Hz)": oscillators(1000, (3, 5))}
spec = lambda X: np.abs(np.fft.rfft(X, axis=1))   # 진폭 스펙트럼 특징 (n, 33)
def tstr_r2(Xg, yg):                               # 생성 자료로 학습(ridge), 실제 자료로 시험
    F, Ft = spec(Xg), spec(X_te)
    w = np.linalg.solve(F.T @ F + 1e-2 * len(F) * np.eye(F.shape[1]), F.T @ (yg - yg.mean()))
    return 1 - np.mean((om_te - Ft @ w - yg.mean()) ** 2) / np.var(om_te)

lag1 = lambda X: np.mean(np.sum(X[:, 1:] * X[:, :-1], 1) / np.sum(X**2, 1))   # 평균 lag-1 자기상관
def peak_hz(X):                                    # 계열마다 가장 강한 주파수의 5–95% 범위, 단위 Hz
    f = (np.argmax(spec(X)[:, 1:], 1) + 1) * 63 / 64  # 표본 간격 1/63 s, 64점 → bin 간격 63/64 Hz
    return f"{np.percentile(f, 5):4.1f}–{np.percentile(f, 95):4.1f}"
print(f"{'':16} 시점별 평균 오차  lag-1 자기상관  주된 주파수(Hz)  TSTR R²(ω)")
print(f"{'실제 자료':16}       -            {lag1(X_te):.3f}        {peak_hz(X_te)}      {tstr_r2(X_real, om_real):.3f}")
for name, (Xg, yg) in gens_ts.items():
    print(f"{name:16}     {np.abs(Xg.mean(0) - X_te.mean(0)).max():.3f}          {lag1(Xg):.3f}        {peak_hz(Xg)}      {tstr_r2(Xg, yg):.3f}")
```

```output
                 시점별 평균 오차  lag-1 자기상관  주된 주파수(Hz)  TSTR R²(ω)
실제 자료                  -            0.795         3.0– 7.9      0.951
좋은 모델                0.039          0.796         3.0– 7.9      0.950
시점별 섞기               0.064          0.004         2.0–30.5      0.224
좁은 주파수(3–5 Hz)       0.039          0.866         3.0– 4.9      -0.392
```

- **시점별 평균은 세 모델 모두 비슷하게 맞는다(0.039–0.064).** 시점별로 섞은 모델은 각 시점의 분포를 정확히 재현하도록 만들었으므로 이 지표로는 결함을 볼 수 없다.
- **lag-1 자기상관이 섞기를 드러낸다(0.795 대 0.004).** 인접한 시점이 서로 다른 계열에서 왔으니 시간 상관이 사라졌다. 이런 계열은 감쇠 진동이 아니라 잡음이다. 주된 주파수가 30 Hz까지 퍼진 것도 같은 결함의 다른 모습이다.
- **좁은 주파수 모델은 개별 표본이 완벽하다.** 물리 방정식을 그대로 따르므로 어떤 물리 검사도 통과한다. 결함은 커버리지에 있고, 주된 주파수의 범위(3.0–4.9 Hz 대 3.0–7.9 Hz)가 그것을 보여 준다.

::quiz{id="marginal-stats-shuffle"}

:::note[분야 예시]
기후 모델 출력을 흉내 내는 생성모델(27.3장)을 평가할 때는 격자점마다의 평균 기온이 맞는지보다 **극값**과 **공간 구조**가 중요하다. 백 년 빈도 강수의 크기, 열파의 지속 기간, 인접 격자 사이의 상관이 실제 모델 출력과 같은지 본다. 시점별·격자별 분포만 맞추고 공간 상관이 깨진 생성 자료로 홍수 위험을 평가하면, 넓은 지역에 동시에 비가 오는 사건의 빈도를 틀리게 추정한다.
:::

## 하위 과제 평가

**왜 필요한가.** 과학 연구에서 생성모델을 쓰는 이유는 대개 생성된 표본 자체가 아니라 그 다음 단계다. 부족한 학습 자료를 늘리거나, 시뮬레이션을 대체하거나, 드문 사건을 더 많이 보기 위해서다. 그렇다면 가장 직접적인 평가는 "생성 데이터가 그 다음 단계에서 실제로 도움이 되는가"다.

**TSTR.** 가장 흔한 형식은 train on synthetic, test on real(TSTR)이다. 생성 자료(와 조건으로 준 레이블)만으로 하위 모델을 학습하고, **실제** 시험 자료에서 성능을 잰다. 실제 자료로 학습한 같은 모델(train on real, test on real)의 성능이 기준선이다. Ravuri와 Vinyals(2019)는 클래스 조건부 이미지 생성모델에 이 방식(Classification Accuracy Score)을 적용해, 생성 이미지만으로 학습한 분류기가 실제 이미지로 학습한 분류기보다 크게 떨어지고, Inception Score나 FID 같은 기존 지표가 이 점수를 예측하지 못한다고 보고했다.

위 표의 마지막 열이 TSTR이다. 생성된 (계열, $\omega$) 쌍으로 진폭 스펙트럼에서 각진동수 $\omega$를 맞히는 ridge 회귀를 학습하고, 실제 시험 계열에서 $R^2$를 쟀다.

- **실제 자료로 학습하면 0.951, 좋은 모델로 학습하면 0.950이다.** 이 과제에서 좋은 모델의 생성 자료는 실제 자료를 대신할 수 있다.
- **시점별로 섞은 모델은 0.224다.** 시간 구조가 깨진 계열로 학습한 모델은 실제 계열에서 진동수를 읽지 못한다.
- **좁은 주파수 모델은 $-0.392$다.** 표본 하나하나는 완벽하지만, 5–8 Hz의 진동을 본 적 없는 회귀 모델이 그 영역에서 엉뚱한 값을 낸다. 평균만 예측해도 0이므로 그보다 나쁘다. 커버리지 실패가 하위 과제에서 가장 크게 드러난다.

**설계할 때 지킬 것.**

1. **하위 모델과 학습 절차를 고정한다.** 생성 자료와 실제 자료에 같은 모델, 같은 하이퍼파라미터 탐색 예산을 쓴다. 예산이 다르면 생성 자료의 효과와 튜닝의 효과가 섞인다(12.5장).
2. **시험 자료는 생성모델의 학습에 쓰지 않은 실제 자료다.** 생성모델이 시험 자료를 본 적이 있으면 기억이 TSTR 점수를 부풀린다.
3. **증강 효과를 따로 잰다.** "실제 자료 + 생성 자료"로 학습한 성능을 "실제 자료만"과 비교하면, 생성 자료가 실제 연구 상황에서 주는 이득을 볼 수 있다. 이득이 0이거나 음수인 경우도 흔하다.
4. **조건부 생성의 레이블을 믿지 않는다.** 조건 $c$로 만든 표본이 정말 $c$에 해당하는지는 따로 확인해야 한다. 레이블이 어긋난 생성 자료는 주변분포 지표가 완벽해도 TSTR을 망친다.

## 실습

위 셀들을 바탕으로, 생성된 시계열을 분포 지표·물리 일관성·하위 과제 유용성으로 평가하는 보고서를 만든다. 생성모델은 19.2장의 자기회귀 모델, 19.5장의 확산 모델, 19.6장의 flow matching 중 직접 구현한 것을 쓴다.

1. **평가 축별 지표를 고른다.** 품질(precision), 커버리지(recall, 주된 주파수 범위), 기억(학습 자료까지 거리 비율), 분포 차이(MMD 검정)를 계산한다. 특징으로는 진폭 스펙트럼처럼 시계열의 구조를 담는 것을 쓰고, 원 시점 값으로 계산한 결과와 비교한다.
2. **도메인 지표를 정의한다.** 감쇠 진동이라면 포락선의 지수 감쇠, 스펙트럼 봉우리의 폭, 자기상관 함수를 실제 자료와 비교한다. 자기 연구 분야의 보존량이나 제약이 있다면 그 위반량을 계산한다.
3. **TSTR과 증강 효과를 잰다.** 실제 자료로 학습한 기준선, 생성 자료만으로 학습한 것, 실제 + 생성으로 학습한 것을 같은 모델·같은 예산으로 비교한다. 실제 학습 자료의 양을 줄여 가며(예: 50, 200, 1000개) 생성 자료의 이득이 어디서 사라지는지 본다.
4. **모든 숫자에 변동을 붙인다.** 생성 표본과 실제 표본을 여러 번 다시 뽑아 지표의 평균과 표준편차를 보고한다. cell 4에서 본 것처럼 FID 하나도 표본에 따라 크게 흔들린다.

이어서 두 연습문제를 푼다. 첫째는 불편 MMD² 추정량을, 둘째는 가우시안 근사의 Fréchet 거리를 구현한다. 테스트가 바로 채점한다.

::exercise{id="mmd-unbiased"}

::exercise{id="frechet-gaussian"}

## 실패 사례

- **한 가지 지표로 생성모델을 평가하는 것.** 이 장의 모든 실험이 이 실패의 예다. 모드를 빠뜨린 모델은 precision이, 흐릿한 모델은 recall이, 복사 모델은 precision·recall·FID가, 시점별로 섞은 모델은 시점별 통계가 좋게 나왔다. 평가 축마다 지표를 따로 두고, 각 지표가 속는 방식을 알고 쓴다.
- **보기 좋은 표본 몇 개를 근거로 삼는 것.** 표본을 골라 보이는 것은 품질에 대한 일화일 뿐이고, 커버리지와 기억에 대해서는 아무것도 말하지 않는다. 그림을 보일 때는 무작위로 뽑은 표본을 보이고, 고른 방법을 밝힌다.
- **학습 자료를 기준으로 분포 지표를 계산하는 것.** 학습 자료를 베끼는 모델이 최고점을 받는다. 기준은 생성모델이 본 적 없는 held-out 실제 자료여야 한다.
- **검정력이 없는 검정의 '유의하지 않음'을 성공으로 보고하는 것.** 표본 수와 kernel 대역폭이 차이를 볼 수 없는 설정이면 어떤 모델도 통과한다. 같은 설정에서 일부러 망가뜨린 모델을 검정해 검정력을 함께 보인다.
- **다른 특징 추출기나 다른 표본 수로 계산한 FID를 비교하는 것.** FID는 특징과 표본 수에 따라 값이 바뀐다. 논문의 FID와 자기 실험의 FID를 비교하려면 같은 구현·같은 표본 수·같은 기준 자료인지 확인한다.

## 통과 기준

- 자신의 생성모델에 대해 품질·커버리지·기억·유용성을 각각 다른 지표로 측정하고, 지표마다 일부러 망가뜨린 대조 모델(모드 누락, 흐림, 복사)에서 그 지표가 실패를 잡는지 확인한 표를 만든다.
- 분포 지표(MMD 또는 FID)를 held-out 실제 자료 기준으로, 반복 추출의 변동과 함께 보고하고, 연습문제 `mmd-unbiased`와 `frechet-gaussian`의 테스트를 통과한다.
- 도메인 지표 하나 이상과 TSTR(실제 자료 기준선 포함)로, 보기 좋은 결과와 분포를 잘 반영한 결과를 구분하는 평가 체계를 제시한다.

## 참고 문헌

- Theis, van den Oord, Bethge (2016). A note on the evaluation of generative models. *International Conference on Learning Representations* (ICLR 2016). arXiv:1511.01844. — 고차원에서 log-likelihood·Parzen 추정·표본 품질이 서로 거의 독립이라는 논증
- Gretton, Borgwardt, Rasch, Schölkopf, Smola (2012). A kernel two-sample test. *Journal of Machine Learning Research*, 13, 723–773. — MMD와 kernel 2-표본 검정
- Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *Advances in Neural Information Processing Systems 30*. arXiv:1706.08500. — FID의 제안
- Kynkäänniemi, Karras, Laine, Lehtinen, Aila (2019). Improved precision and recall metric for assessing generative models. *Advances in Neural Information Processing Systems 32*. arXiv:1904.06991. — k-NN 다양체 근사로 품질과 커버리지를 따로 재는 precision·recall
- Ravuri, Vinyals (2019). Classification accuracy score for conditional generative models. *Advances in Neural Information Processing Systems 32*. arXiv:1905.10887. — 생성 자료로 학습하고 실제 자료로 시험하는 하위 과제 평가
- Carlini, Hayes, Nasr, Jagielski, Sehwag, Tramèr, Balle, Ippolito, Wallace (2023). Extracting training data from diffusion models. *32nd USENIX Security Symposium*, 5253–5270. — 확산 모델의 학습 자료 기억과 추출
