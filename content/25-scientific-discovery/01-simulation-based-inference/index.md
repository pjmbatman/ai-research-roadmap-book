---
id: scientific-discovery/simulation-based-inference
title: "시뮬레이션 기반 추론"
title_en: "Simulation-Based Inference"
tier: advanced
status: draft
prerequisites: [scientific-ml/inverse-problems-and-uq, generative-and-foundation-models/generative-modeling]
---

# 시뮬레이션 기반 추론

과학의 많은 모델은 식이 아니라 **프로그램**이다. 유전자 계보를 무작위로 그리는 집단유전학 시뮬레이터, 은하 분포를 만드는 우주론 시뮬레이션, 감염이 사람 사이를 무작위로 건너가는 행위자 기반 역학 모델, 검출기 안에서 입자가 부딪히는 과정을 따라가는 몬테카를로 코드가 그렇다. 파라미터 $\theta$를 넣으면 그럴듯한 데이터 $\mathbf{x}$를 **뽑아** 주지만, "이 파라미터에서 관측값 $\mathbf{x}_{\text{obs}}$가 나올 확률밀도는 얼마인가"는 답하지 못한다. 13장의 베이지안 추론은 모두 그 확률밀도, 즉 우도를 계산할 수 있다는 가정에서 출발했다.

이 장은 우도 없이 시뮬레이터만으로 사후분포 $p(\theta \mid \mathbf{x}_{\text{obs}})$를 구하는 방법을 다룬다. 고전적인 ABC(approximate Bayesian computation)에서 시작해 신경망으로 사후분포를 직접 학습하는 방법으로 넘어가고, 마지막으로 어떤 방법을 쓰든 반드시 해야 하는 점검, 즉 사후분포가 **보정되어 있는지**와 **시뮬레이터가 현실과 맞는지**를 확인하는 법을 본다.

## 학습 목표

- 우도를 쓸 수 없는 시뮬레이터로 파라미터를 추론한다
- 신경망 기반 사후 추정을 평가한다
- 사후분포의 보정을 coverage로 점검하고 시뮬레이터 오차를 진단한다

## 우도 없는 추론

**왜 필요한가.** 확률적 시뮬레이터 안에는 관측되지 않는 무작위 경로 $\mathbf{z}$(유전자 계보, 개별 감염 사건, 입자의 충돌 순서)가 있다. 우도는 그 경로를 모두 적분한 값이다.

$$
p(\mathbf{x} \mid \theta) = \int p(\mathbf{x}, \mathbf{z} \mid \theta)\, d\mathbf{z}.
$$

$\mathbf{z}$가 수천–수백만 차원이면 이 적분은 계산할 수 없다. 반면 $\mathbf{z}$를 하나 뽑고 그 경로를 따라 $\mathbf{x}$를 만드는 일은 시뮬레이터가 늘 하는 일이다. 이런 모델을 **암묵적 모델**(implicit model)이라 하고, 표본만 쓰는 추론을 **우도 없는 추론**(likelihood-free inference) 또는 **시뮬레이션 기반 추론**(simulation-based inference, SBI)이라 한다.

**정의.** 목표는 여전히 베이즈 정리의 사후분포다.

$$
p(\theta \mid \mathbf{x}_{\text{obs}}) \propto p(\mathbf{x}_{\text{obs}} \mid \theta)\, p(\theta).
$$

달라진 것은 쓸 수 있는 연산이다. 사전분포 $p(\theta)$에서 뽑을 수 있고, 시뮬레이터로 $\mathbf{x} \sim p(\mathbf{x} \mid \theta)$를 뽑을 수 있다. 우도의 값은 계산할 수 없다. 그래서 MCMC(13.5장)의 수용 확률처럼 우도 비를 요구하는 알고리즘은 그대로 쓸 수 없다. 24.7장의 베이지안 역문제가 "결정론적 순방향 모델 + 가우시안 측정 잡음"으로 우도를 직접 썼다면, 이 장은 순방향 모델 자체가 확률적이어서 우도를 쓸 수 없는 경우다.

**직관.** 사전분포에서 $(\theta, \mathbf{x})$ 쌍을 많이 뽑으면 결합분포 $p(\theta, \mathbf{x})$의 표본이 된다. 사후분포는 그 결합분포를 $\mathbf{x} = \mathbf{x}_{\text{obs}}$에서 자른 단면이다. 모든 SBI 방법은 "결합분포의 표본에서 단면을 어떻게 근사하는가"의 변형이다. ABC는 $\mathbf{x}_{\text{obs}}$ 근처의 표본만 남겨 단면을 근사하고, 신경 사후 추정은 결합 표본으로 조건부분포 $p(\theta \mid \mathbf{x})$를 회귀처럼 학습한다. 19.1장의 관점으로 보면, 신경 사후 추정은 $\mathbf{x}$를 조건으로 하는 $\theta$의 조건부 생성모델이다.

이 장 전체에서 쓰는 예는 1차 자기회귀 과정 AR(1)이다.

$$
x_t = \rho\, x_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1), \qquad x_0 = 0, \quad t = 1, \dots, 50.
$$

$\rho$는 과정의 "기억"이다. 0이면 매 시각이 독립인 잡음이고, 1에 가까우면 한 번 벗어난 값이 오래 유지된다. 기온 편차, 지하수위, 세포 안 단백질 농도의 요동처럼 관성이 있는 잡음을 가장 단순하게 흉내 낸 모델이다. 이 모델은 우도를 계산할 수 있다. 그래서 일부러 골랐다. 추론 알고리즘에는 시뮬레이터로만 쓰고, 정확한 사후분포는 채점용 정답으로만 쓴다. 사전분포는 $\rho \sim U(0, 1)$, 참값은 $\rho = 0.6$이다. 시뮬레이션 2만 번을 미리 돌려 **참조 테이블**(reference table)로 저장해 둔다.

```python {exec="browser" session="sbi"}
import numpy as np

T = 50                                            # 시계열 길이
def simulate(rho, rng):                           # rho: (N,) -> x: (N, T)
    x = np.zeros((len(rho), T))
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + rng.normal(size=len(rho))
    return x

def summaries(x):                                 # (N, T) -> (N, 3): 평균, 분산, lag-1 자기상관
    xc = x - x.mean(1, keepdims=True)
    ac1 = (xc[:, 1:] * xc[:, :-1]).sum(1) / (xc**2).sum(1)
    return np.stack([x.mean(1), x.var(1), ac1], axis=1)

rng = np.random.default_rng(1)
rho_true = 0.6
x_obs = simulate(np.array([rho_true]), rng)       # (1, T) 관측 데이터
N = 20000
rho_sim = rng.uniform(0, 1, N)                    # 사전분포 U(0, 1)에서 뽑은 파라미터
x_sim = simulate(rho_sim, rng)                    # (N, T) 참조 테이블: 시뮬레이션 2만 번
S_sim = summaries(x_sim)                          # (N, 3)
s_obs = summaries(x_obs)[0]                       # (3,)

# 비교 기준: 이 장난감 모델은 우도를 계산할 수 있으므로 격자로 정확한 사후분포를 구한다
grid = np.linspace(0, 1, 2001)
loglik = -0.5 * ((x_obs[0, 1:] - grid[:, None] * x_obs[0, :-1]) ** 2).sum(1)
post = np.exp(loglik - loglik.max()); post /= post.sum()
m_exact = (grid * post).sum(); sd_exact = np.sqrt(((grid - m_exact) ** 2 * post).sum())
print(f"관측 요약 통계 평균 {s_obs[0]:.3f}  분산 {s_obs[1]:.3f}  자기상관 {s_obs[2]:.3f}")
print(f"정확한 사후분포  평균 {m_exact:.3f}  표준편차 {sd_exact:.3f}")
```

```output
관측 요약 통계 평균 -0.140  분산 1.093  자기상관 0.554
정확한 사후분포  평균 0.563  표준편차 0.134
```

관측 데이터 한 개의 정확한 사후 평균은 0.563으로 참값 0.6과 조금 다르다. 시계열 50개로는 이 정도 불확실성(표준편차 0.134)이 남는다. 이것이 이 장의 모든 근사 방법이 맞혀야 할 기준이다. 참값이 아니라 **정확한 사후분포**와 비교한다는 점에 주의한다. 사후분포가 참값을 벗어나는 것은 근사의 잘못이 아니라 데이터가 유한하기 때문이다.

## 고전적 방법

**ABC rejection.** 가장 오래된 SBI 방법은 거르기다. 사전분포에서 $\theta_i$를 뽑고, 시뮬레이터로 $\mathbf{x}_i$를 만들고, $\mathbf{x}_i$가 관측값과 충분히 가까우면 $\theta_i$를 받아들인다. 받아들인 $\theta_i$들이 사후분포의 근사 표본이다. 이 알고리즘은 Pritchard 외(1999)가 집단유전학에서 Y 염색체 데이터로 인구 성장을 추론할 때 쓴 형태로 널리 알려졌다.

"충분히 가깝다"를 정하려면 두 가지를 골라야 한다. 시계열 50개를 통째로 비교하면 거의 아무것도 가깝지 않으므로, 먼저 데이터를 몇 개의 **요약 통계**(summary statistic) $\mathbf{s}(\mathbf{x})$로 줄인다. 그다음 거리 $d$와 **허용 오차**(tolerance) $\epsilon$을 정한다. ABC가 실제로 표본을 뽑는 분포는

$$
p_\epsilon(\theta \mid \mathbf{x}_{\text{obs}}) \propto p(\theta)\, \Pr\big[d(\mathbf{s}(\mathbf{x}), \mathbf{s}(\mathbf{x}_{\text{obs}})) \le \epsilon \mid \theta\big]
$$

이다. 두 극한이 중요하다. $\epsilon \to \infty$이면 모든 표본을 받아들이므로 $p_\epsilon$은 **사전분포**가 된다. $\epsilon \to 0$이면 $p(\theta \mid \mathbf{s}_{\text{obs}})$, 즉 요약 통계만 보았을 때의 사후분포가 된다. 이것이 $p(\theta \mid \mathbf{x}_{\text{obs}})$와 같으려면 요약 통계가 $\theta$에 대해 **충분 통계량**(sufficient statistic)이어야 한다. 실제 시뮬레이터에서 충분 통계량을 아는 경우는 드물다. 그래서 ABC의 오차는 두 겹이다. 요약 통계가 정보를 버린 오차, 그리고 $\epsilon$이 0이 아니어서 생기는 오차다.

실무에서는 $\epsilon$ 대신 **수용 비율** $q$를 정하고, 거리가 가장 가까운 비율 $q$의 시뮬레이션을 받아들인다. 통계마다 척도가 다르므로 거리를 재기 전에 각 통계를 표준편차로 나눈다. 요약 통계 세 개(평균, 분산, lag-1 자기상관)를 하나씩, 그리고 모두 함께 써 본다.

```python {exec="browser" session="sbi"}
def abc(S_sim, s_obs, cols, q=0.01):
    """요약 통계 열 cols로 거리를 재고, 가장 가까운 비율 q의 시뮬레이션을 받아들인다."""
    scale = S_sim[:, cols].std(0)                 # 통계마다 척도가 달라 표준화한다
    d = np.sqrt((((S_sim[:, cols] - s_obs[cols]) / scale) ** 2).sum(1))   # (N,)
    return d <= np.quantile(d, q)                 # (N,) 수용 여부

for name, cols in [("평균", [0]), ("분산", [1]), ("자기상관", [2]), ("셋 다", [0, 1, 2])]:
    acc = abc(S_sim, s_obs, cols)
    r = rho_sim[acc]
    print(f"{name:6s} 수용 {acc.sum()}개  사후 평균 {r.mean():.3f}  표준편차 {r.std():.3f}")
for q in [0.001, 0.01, 0.1, 0.5]:
    r = rho_sim[abc(S_sim, s_obs, [2], q)]
    print(f"자기상관, 수용 비율 {q:5.3f}: 사후 평균 {r.mean():.3f}  표준편차 {r.std():.3f}")
```

```output
평균     수용 200개  사후 평균 0.408  표준편차 0.240
분산     수용 200개  사후 평균 0.327  표준편차 0.208
자기상관   수용 200개  사후 평균 0.605  표준편차 0.130
셋 다    수용 200개  사후 평균 0.569  표준편차 0.138
자기상관, 수용 비율 0.001: 사후 평균 0.636  표준편차 0.157
자기상관, 수용 비율 0.010: 사후 평균 0.605  표준편차 0.130
자기상관, 수용 비율 0.100: 사후 평균 0.613  표준편차 0.136
자기상관, 수용 비율 0.500: 사후 평균 0.608  표준편차 0.190
```

결과를 정확한 사후분포(평균 0.563, 표준편차 0.134)와 비교한다.

- **평균만 쓰면** 사후 표준편차가 0.240으로, 사전분포 $U(0, 1)$의 표준편차 $1/\sqrt{12} \approx 0.289$와 크게 다르지 않다. AR(1)의 평균은 $\rho$와 상관없이 0 근처이므로 $\rho$에 대한 정보가 거의 없다.
- **분산만 쓰면** 사후 평균이 0.327로 치우친다. 정상 상태 분산 $1/(1 - \rho^2)$은 $\rho$에 따라 변하지만, 시계열 50개로 잰 분산은 잡음이 커서 이번 관측(1.093)처럼 작게 나오면 $\rho$를 작게 본다.
- **lag-1 자기상관만 쓰면** 정확한 결과에 가장 가깝다. 이 모델의 로그우도는 데이터에 $\sum_t x_t x_{t-1}$과 $\sum_t x_{t-1}^2$를 통해서만 의존하고, lag-1 자기상관은 그 비, 즉 최대우도 추정값과 거의 같은 양이다. 거의 충분한 통계량이다.
- **셋 다 쓰면** 자기상관만 쓴 결과와 다르다. 정보가 없는 통계(평균)도 거리 계산에 같은 무게로 들어가 "가까운" 표본을 고르는 기준을 흐린다. 통계를 늘린다고 늘 나아지지 않는다.

::quiz{id="uninformative-summary"}

수용 비율을 바꾸면 두 번째 오차가 보인다. $q = 0.5$이면 표준편차가 0.190으로 넓어진다. 사전분포 쪽으로 끌려가는 것이다. 반대로 $q = 0.001$은 2만 개 중 20개만 받아들여, 근사 편향은 작지만 표본이 적어 추정값 자체가 흔들린다(표준편차 0.157). 같은 시뮬레이션 예산에서 편향과 분산의 균형점이 있다.

요약 통계가 $k$개이면 관측값 근처 반지름 $\epsilon$의 공에 들어갈 확률은 대략 $\epsilon^k$에 비례한다. 통계가 늘수록 같은 $\epsilon$에서 받아들여지는 표본이 기하급수적으로 줄어든다. ABC가 저차원 요약 통계에 의존하는 이유다. 이 한계를 줄이려는 개선이 여러 갈래로 있다. 받아들인 표본을 요약 통계에 대한 국소 선형 회귀로 보정하는 방법(Beaumont 외, 2002), 허용 오차를 줄여 가며 표본을 이어서 개선하는 순차 몬테카를로 ABC, 그리고 다음 절의 신경망 기반 방법이다.

:::note[분야 예시]
ABC는 집단유전학에서 자랐다. 인구 크기의 변화나 이동 이력을 파라미터로 두고 유전자 계보를 시뮬레이션하면, 관측된 DNA 다양성의 우도는 가능한 계보 전체에 대한 적분이라 계산할 수 없다. 그래서 다양성 지표 몇 개를 요약 통계로 두고 시뮬레이션과 비교했다(Pritchard 외, 1999; Beaumont 외, 2002). 같은 구조가 감염병 전파 모델(전파율과 회복률을 발생 곡선의 요약 통계로 추론), 생태학의 개체군 동역학, 우주론의 구조 형성 시뮬레이션에도 나타난다.
:::

::widget{name="abc-tolerance"}

## 신경망 기반 추론

**왜 필요한가.** ABC는 관측값이 바뀔 때마다 처음부터 거르기를 다시 하고, 요약 통계를 사람이 골라야 하며, 통계 차원이 늘면 수용률이 무너진다. 신경망은 이 세 가지를 각각 다른 방식으로 완화한다.

**신경 사후 추정**(neural posterior estimation, NPE). 결합분포에서 뽑은 쌍 $(\theta_i, \mathbf{x}_i)$로 조건부 밀도 모델 $q_\phi(\theta \mid \mathbf{x})$를 학습한다. 손실은 음의 로그우도다.

$$
L(\phi) = -\mathbb{E}_{p(\theta, \mathbf{x})}\big[\log q_\phi(\theta \mid \mathbf{x})\big]
= \mathbb{E}_{p(\mathbf{x})}\Big[\mathrm{KL}\big(p(\theta \mid \mathbf{x}) \,\|\, q_\phi(\theta \mid \mathbf{x})\big)\Big] + \text{상수}.
$$

두 번째 등호는 $\log q = \log p(\theta \mid \mathbf{x}) - \log\frac{p(\theta \mid \mathbf{x})}{q}$로 나누어 기댓값을 취하면 나온다. 상수는 $\phi$와 무관한 $-\mathbb{E}[\log p(\theta \mid \mathbf{x})]$다. 그러니 이 손실을 최소화하면, 사전 예측분포 $p(\mathbf{x})$가 자주 내는 데이터에서 $q_\phi$가 참 사후분포에 가까워진다. 우도는 어디에도 필요 없다. 시뮬레이터가 만든 쌍만 있으면 된다. $q_\phi$로는 가우시안 혼합(mixture density network)이나 normalizing flow(19.4장)를 흔히 쓴다. Papamakarios와 Murray(2016)는 혼합 밀도 신경망으로 사후분포를 직접 학습해 ABC의 허용 오차 $\epsilon$을 없애는 방법을 제안했다.

**신경 우도 추정**(neural likelihood estimation, NLE)은 반대 방향 $q_\phi(\mathbf{x} \mid \theta)$를 학습한 뒤, 그것을 우도 대신 넣고 MCMC(13.5장)로 사후분포를 뽑는다. **신경 비율 추정**(neural ratio estimation, NRE)은 같은 시뮬레이션에서 나온 쌍 $(\theta, \mathbf{x})$와 서로 섞은 쌍을 구별하는 분류기를 학습한다. 최적 분류기의 로짓이 $\log \frac{p(\mathbf{x} \mid \theta)}{p(\mathbf{x})}$이므로, 이 비율에 사전분포를 곱하면 사후분포가 된다. 세 방법 모두 "시뮬레이터 표본으로 밀도나 비율을 학습한다"는 점이 같고, 무엇을 학습하느냐가 다르다. 전체 지형은 Cranmer 외(2020)의 리뷰가 정리한다.

**Amortization.** NPE를 한 번 학습하면 새 관측값에는 신경망의 순방향 계산 한 번으로 사후분포를 얻는다. 추론 비용을 학습 단계에 미리 치르는 것을 **amortized inference**라 한다. 같은 실험을 많이 반복하거나(검출기 사건 수백만 개), 같은 모델을 여러 관측소·환자·시료에 적용할 때 유리하다. 반대로 관측값이 하나뿐이면, 그 관측값 근처에 시뮬레이션을 집중하는 **순차적**(sequential) 방법이 같은 예산으로 더 정확할 수 있다. 대신 순차적 방법은 다른 관측값에 다시 쓸 수 없다.

짧은 예로 NPE를 만든다. 밀도 모델은 가장 단순한 것을 쓴다. $\rho$의 범위 $[0, 1]$을 구간 20개로 나누고, 신경망 분류기가 "이 데이터의 $\rho$가 어느 구간에 있는가"의 확률을 출력하게 한다. 교차 엔트로피 손실은 위 식의 음의 로그우도를 구간 히스토그램이라는 밀도 모델에 적용한 것과 같다. 입력은 사람이 고른 요약 통계 하나 대신 lag 1–5의 자기상관과 평균·로그 분산을 넣어, 어떤 조합이 유용한지는 신경망이 정하게 한다.

```python {exec="browser" session="sbi"}
def coverage(cols, q, M=300, level=0.9, seed=1):
    rng = np.random.default_rng(seed)
    rho_test = rng.uniform(0, 1, M)                # 사전분포에서 뽑은 '참값'
    S_test = summaries(simulate(rho_test, rng))   # (M, 3)
    hit, width = 0, 0.0
    for m in range(M):                            # 참조 테이블은 다시 쓴다(시뮬레이션 추가 없음)
        r = rho_sim[abc(S_sim, S_test[m], cols, q)]
        lo, hi = np.quantile(r, [(1 - level) / 2, (1 + level) / 2])
        hit += lo <= rho_test[m] <= hi
        width += hi - lo
    return hit / M, width / M

for cols, q in [([2], 0.01), ([2], 0.2), ([2], 0.8)]:
    cov, w = coverage(cols, q)
    print(f"통계 {cols}, q = {q:4.2f}: 90% 구간 coverage {cov:.2f}  평균 폭 {w:.3f}")
```

```output
통계 [2], q = 0.01: 90% 구간 coverage 0.88  평균 폭 0.374
통계 [2], q = 0.20: 90% 구간 coverage 0.90  평균 폭 0.425
통계 [2], q = 0.80: 90% 구간 coverage 0.90  평균 폭 0.788
```

학습은 셀 1의 참조 테이블을 그대로 썼고, 새 데이터 300개의 사후 구간은 재학습 없이 순방향 계산만으로 얻었다. 관측 데이터의 90% 구간은 정확한 사후분포의 구간과 거의 겹치고, 시험 데이터 300개에서 90% 구간이 참값을 포함한 비율도 0.91로 명목값 0.9에 가깝다. 실제 연구에서는 구간 20개의 히스토그램 대신 flow를 쓰고, 요약 특징도 시계열 자체를 받는 신경망(1차원 CNN, Transformer)이 학습하게 한다. 그만큼 시뮬레이션과 학습이 더 필요하고, 다음 절의 점검이 더 중요해진다.

::quiz{id="npe-amortization"}

## 진단

**왜 필요한가.** 근사 사후분포는 틀려도 겉으로는 멀쩡해 보인다. 평균과 표준편차가 그럴듯한 숫자로 나오고, 오류 메시지도 없다. 무엇이 틀렸는지 알려 주는 것은 별도의 점검뿐이다. 점검은 두 질문으로 나뉜다. 추론 알고리즘이 **모델 안에서** 맞는가(보정), 그리고 **모델이 현실과** 맞는가(시뮬레이터 오차)다.

**보정과 coverage.** 시뮬레이터가 맞다고 가정하면 점검용 정답을 얼마든지 만들 수 있다. 사전분포에서 참값 $\tilde\theta$를 뽑고, 그 값으로 데이터 $\tilde{\mathbf{x}}$를 시뮬레이션하고, 그 데이터로 추론해 수준 $1 - \alpha$의 사후 구간을 만든다. 추론이 정확하면 이 구간은 평균적으로 비율 $1 - \alpha$만큼 참값을 포함한다.

$$
\Pr_{\tilde\theta \sim p(\theta),\ \tilde{\mathbf{x}} \sim p(\mathbf{x} \mid \tilde\theta)}\big[\tilde\theta \in C_{1-\alpha}(\tilde{\mathbf{x}})\big] = 1 - \alpha.
$$

이 비율을 **경험적 coverage**라 한다. 명목값보다 낮으면 사후분포가 너무 좁다(**과신**, overconfident). 높으면 너무 넓다(**보수적**, conservative). 구간 대신 참값의 사후 표본 내 순위를 보는 것이 13.8장의 SBC(simulation-based calibration)이고(Talts 외, 2018), 같은 원리다. Hermans 외(2022)는 여러 SBI 알고리즘을 벤치마크해, 신경망 기반 방법과 ABC 모두 과신한 사후분포를 낼 수 있음을 보이고 coverage 점검을 기본 절차로 권했다.

ABC로 coverage를 재 본다. 참조 테이블을 다시 쓰므로 시뮬레이션을 추가로 돌릴 필요가 없다. 함께 **평균 구간 폭**을 출력한다.

```python {exec="browser" session="sbi"}
import warnings
from sklearn.neural_network import MLPClassifier
warnings.simplefilter("ignore")                       # 수렴 경고는 숨긴다

def features(x, lags=5):                              # (N, T) -> (N, lags + 2): 정보를 덜 버리는 넉넉한 특징
    xc = x - x.mean(1, keepdims=True)
    ac = [(xc[:, k:] * xc[:, :-k]).sum(1) / (xc**2).sum(1) for k in range(1, lags + 1)]
    return np.stack(ac + [x.mean(1), np.log(x.var(1))], axis=1)

K = 20                                                # 사후분포를 [0, 1]의 구간 20개 위 히스토그램으로 표현
net = MLPClassifier(hidden_layer_sizes=(32,), max_iter=30, random_state=0)
net.fit(features(x_sim), np.minimum((rho_sim * K).astype(int), K - 1))   # 입력 x -> rho가 속한 구간

def npe_interval(x, level=0.9):                       # (M, T) -> 구간 하한·상한 (M,), (M,)
    cdf = np.cumsum(net.predict_proba(features(x)), axis=1)   # (M, K)
    edges = np.arange(1, K + 1) / K                   # 각 구간의 오른쪽 끝
    lo = np.array([np.interp((1 - level) / 2, np.r_[0, c], np.r_[0, edges]) for c in cdf])
    hi = np.array([np.interp((1 + level) / 2, np.r_[0, c], np.r_[0, edges]) for c in cdf])
    return lo, hi

lo, hi = npe_interval(x_obs)
cdf_exact = np.cumsum(post)                           # 셀 1의 격자 사후분포
print(f"NPE 90% 구간   [{lo[0]:.2f}, {hi[0]:.2f}]")
print(f"정확한 90% 구간 [{np.interp(0.05, cdf_exact, grid):.2f}, {np.interp(0.95, cdf_exact, grid):.2f}]")
rng_t = np.random.default_rng(1)
rho_test = rng_t.uniform(0, 1, 300)
lo, hi = npe_interval(simulate(rho_test, rng_t))      # 새 데이터 300개에 재학습 없이 적용
print(f"NPE 90% 구간 coverage {np.mean((lo <= rho_test) & (rho_test <= hi)):.2f}  평균 폭 {np.mean(hi - lo):.2f}")
```

```output
NPE 90% 구간   [0.32, 0.75]
정확한 90% 구간 [0.34, 0.78]
NPE 90% 구간 coverage 0.91  평균 폭 0.38
```

세 설정 모두 coverage가 0.9 근처다. 그런데 구간 폭은 0.374에서 0.788로 두 배가 넘게 차이 난다. $q = 0.8$의 ABC는 거의 사전분포를 돌려주고 있지만 coverage 점검은 통과한다. 사전분포 자체가 완벽하게 보정된 "사후분포"이기 때문이다. 사전분포에서 참값을 뽑았으니, 사전분포의 90% 구간은 정확히 90%의 참값을 포함한다. 그러므로 **coverage는 필요조건이지 충분조건이 아니다.** 보정을 확인한 다음에는 구간이 얼마나 좁은지(sharpness), 가능하면 정확한 방법이나 더 많은 시뮬레이션을 쓴 기준과 비교해 정보를 얼마나 잃었는지를 함께 보고한다. 21.2장의 "보정된 범위 안에서 가장 날카롭게"와 같은 원칙이다.

::quiz{id="prior-interval-width"}

**시뮬레이터 오차**(model misspecification). coverage 점검은 시뮬레이터가 현실을 정확히 재현한다고 가정한 채로 하는 점검이다. 실제 데이터가 시뮬레이터에 없는 과정을 거쳐 나왔다면 이 점검은 아무것도 말해 주지 않는다. 가장 흔한 예가 측정 장비의 잡음이다. AR(1) 과정을 표준편차 1의 측정 잡음과 함께 관측했는데 시뮬레이터에는 그 잡음이 없다고 하자. 측정 잡음은 이웃한 시각 사이의 상관을 희석하므로, 자기상관으로 추론하면 $\rho$를 체계적으로 작게 본다.

이런 오류를 잡는 도구는 **사후 예측 점검**(posterior predictive check, 13.8장)이다. 받아들인 파라미터로 시뮬레이션한 데이터가 관측 데이터를 닮았는지 본다. 특히 **추론에 쓰지 않은** 통계로 비교해야 한다. 추론에 쓴 통계는 맞추도록 골랐으니 당연히 맞는다. 여기서는 자기상관으로 추론하고 분산으로 점검한다. 받아들인 시뮬레이션의 분산 가운데 관측 분산 이상인 비율 $p$를 계산하고, $p < 0.05$이면 "관측 분산이 모델이 예측한 범위보다 크다"는 경고로 본다.

```python {exec="browser" session="sbi"}
rng = np.random.default_rng(2)
rho_test = rng.uniform(0, 1, 300)
x_clean = simulate(rho_test, rng)                     # (300, T) 시뮬레이터와 같은 과정
x_noisy = x_clean + rng.normal(size=x_clean.shape)    # 시뮬레이터에 없는 측정 잡음이 더해진 장비

for name, x in [("시뮬레이터와 같은 데이터", x_clean), ("측정 잡음이 섞인 데이터", x_noisy)]:
    S = summaries(x)
    hit, p = 0, []
    for m in range(len(S)):
        acc = abc(S_sim, S[m], [2])                   # 자기상관으로 추론
        lo, hi = np.quantile(rho_sim[acc], [0.05, 0.95])
        hit += lo <= rho_test[m] <= hi
        p.append(np.mean(S_sim[acc, 1] >= S[m, 1]))   # 추론에 쓰지 않은 분산으로 사후 예측 점검
    print(f"{name}: 90% coverage {hit / len(S):.2f}   분산 점검 p < 0.05 비율 {np.mean(np.array(p) < 0.05):.2f}")
```

```output
시뮬레이터와 같은 데이터: 90% coverage 0.89   분산 점검 p < 0.05 비율 0.04
측정 잡음이 섞인 데이터: 90% coverage 0.63   분산 점검 p < 0.05 비율 0.98
```

시뮬레이터와 같은 과정에서 나온 데이터에서는 coverage가 0.89이고, 분산 점검 경고는 4%로 거짓 경보 수준(약 5%)에 머문다. 측정 잡음이 섞인 데이터에서는 90% 구간이 참값을 63%만 포함한다. 사후분포가 틀린 곳에 자신 있게 모여 있다. 그리고 사후분포만 보아서는 이를 알 수 없다. 구간 폭은 멀쩡하기 때문이다. 같은 데이터에서 분산 점검은 98%의 경우에 경고를 냈다. 관측 분산이 받아들인 $\rho$로는 설명되지 않을 만큼 크다는 신호다. 경고가 뜨면 사후분포를 보고하기 전에 시뮬레이터를 고친다. 이 예에서는 측정 잡음 항을 시뮬레이터에 넣고, 잡음 크기도 추론할 파라미터에 추가하는 것이 해결책이다.

::quiz{id="low-coverage-meaning"}

:::note[분야 예시]
입자물리에서는 검출기 시뮬레이션이 실제 검출기 반응을 완벽하게 재현하지 못한다. 그래서 시뮬레이션으로 학습한 추론기를 실제 데이터에 쓰기 전에, 결과가 이미 알려진 대조 영역(control region)의 데이터에서 시뮬레이션과 실제 분포를 비교한다. 계산 신경과학에서 뉴런 모델의 파라미터를 전압 기록으로 추론할 때나, 기후 모델의 파라미터를 관측으로 보정할 때도 같은 원칙이 적용된다. 시뮬레이터가 재현하지 못하는 특징이 있으면 그 특징은 추론에 쓰지 않고, 쓰지 않은 특징으로 모델을 점검한다.
:::

## 실습

본문의 셀 다섯 개는 `labs/scientific-discovery/simulation-based-inference/cells.py`에 그대로 있다. 다음 순서로 확장한다.

1. 셀 2에서 참조 테이블 크기 $N$을 2000, 20000, 200000으로 바꾸고 수용 비율을 고정했을 때 사후 표준편차와 정확한 사후분포의 차이가 어떻게 줄어드는지 기록한다.
2. 셀 3의 coverage를 50%, 80%, 95% 수준에서도 계산해, 수준별 명목값과 경험값을 한 표로 정리한다. 모든 수준에서 맞아야 보정되었다고 말할 수 있다.
3. 셀 4의 NPE를 학습 데이터 2000개로 다시 학습하고 coverage가 어떻게 바뀌는지 본다. 학습 데이터가 부족한 신경 사후 추정이 과신하는지 보수적인지를 기록한다.
4. 셀 5의 시뮬레이터에 측정 잡음 표준편차 $\sigma_{\text{obs}}$를 두 번째 파라미터로 넣고, $(\rho, \sigma_{\text{obs}})$를 함께 추론해 coverage가 회복되는지 확인한다.

두 연습문제에서 ABC 수용 단계와 coverage 계산을 직접 구현한다. 테스트가 바로 채점한다.

::exercise{id="abc-rejection"}

::exercise{id="interval-coverage"}

## 실패 사례

- **시뮬레이터가 현실과 다를 때 사후분포를 그대로 믿는 것.** coverage 점검은 시뮬레이터 안에서의 점검이라 이 오류를 잡지 못한다. 셀 5처럼 측정 잡음 하나가 빠져도 90% 구간이 참값을 63%만 포함한다. 추론에 쓰지 않은 통계로 사후 예측 점검을 하고, 경고가 뜨면 시뮬레이터를 고친다.
- **coverage가 맞으니 좋은 사후분포라고 결론 내리는 것.** 사전분포도 coverage를 통과한다. 구간 폭과 함께, 가능하면 더 정확한 기준과 비교해 보고한다.
- **요약 통계를 많이 넣을수록 낫다고 생각하는 것.** 정보 없는 통계는 거리 계산을 흐리고, 통계 차원이 늘수록 같은 허용 오차에서 받아들이는 표본이 급격히 줄어든다.
- **관측 데이터 하나로 방법을 비교하는 것.** 한 데이터에서 정확한 사후분포에 더 가까웠던 방법이 다른 데이터에서는 아닐 수 있다. 셀 3처럼 사전분포에서 뽑은 시험 데이터 수백 개로 평균적인 성질을 본다.
- **사전분포의 범위가 결과를 정한다는 것을 잊는 것.** 이 장의 모든 방법은 사전분포에서 뽑은 시뮬레이션만 본다. 참값이 사전분포 범위 밖이면 어떤 방법도 그 값을 찾지 못한다.

## 통과 기준

- 우도 없이 시뮬레이터만으로 ABC와 신경 사후 추정을 각각 실행하고, 결과를 정확한 사후분포(또는 더 많은 시뮬레이션을 쓴 기준)와 비교해 보고한다.
- 추론 결과를 여러 수준의 coverage와 평균 구간 폭, 그리고 추론에 쓰지 않은 통계의 사후 예측 점검과 함께 보고한다.
- 연습문제 두 개(`abc-rejection`, `interval-coverage`)의 테스트를 통과한다.

## 참고 문헌

- Cranmer, Brehmer, Louppe (2020). The frontier of simulation-based inference. *Proceedings of the National Academy of Sciences*, 117(48), 30055–30062. — SBI 방법(ABC, 신경 사후·우도·비율 추정)의 전체 지형
- Pritchard, Seielstad, Perez-Lezaun, Feldman (1999). Population growth of human Y chromosomes: a study of Y chromosome microsatellites. *Molecular Biology and Evolution*, 16(12), 1791–1798. — 요약 통계를 쓰는 rejection ABC의 초기 적용
- Beaumont, Zhang, Balding (2002). Approximate Bayesian computation in population genetics. *Genetics*, 162(4), 2025–2035. — 요약 통계에 대한 국소 선형 회귀 보정
- Papamakarios, Murray (2016). Fast ε-free inference of simulation models with Bayesian conditional density estimation. *NeurIPS 29*. arXiv:1605.06376. — 혼합 밀도 신경망으로 사후분포를 직접 학습
- Hermans, Delaunoy, Rozet, Wehenkel, Begy, Louppe (2022). A trust crisis in simulation-based inference? Your posterior approximations can be unfaithful. arXiv:2110.06581. — SBI 알고리즘의 과신과 coverage 점검
- Talts, Betancourt, Simpson, Vehtari, Gelman (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv:1804.06788. — SBC의 순위 통계량
