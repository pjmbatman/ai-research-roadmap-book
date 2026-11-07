---
id: scientific-discovery/autonomous-experimentation
title: "자율 실험과 closed-loop 연구"
title_en: "Autonomous Experimentation and Closed-Loop Research"
tier: advanced
status: draft
prerequisites: [scientific-discovery/experimental-design]
---

# 자율 실험과 closed-loop 연구

25.3장은 다음 실험을 고르는 기준을 다뤘다. 실제 연구에서는 그 기준이 혼자 쓰이지 않는다. 모델이 실험을 고르고, 장비가 실험을 수행하고, 측정 결과가 모델을 갱신하고, 갱신된 모델이 다시 실험을 고른다. 이 고리를 사람의 개입 없이(또는 최소한의 개입으로) 반복하는 것이 **closed-loop** 연구이고, 그것을 물리적 실험 장비로 구현한 것이 **자율 실험실**(self-driving lab)이다.

고리를 닫으면 실험 속도가 빨라지는 만큼 실패도 빨라진다. 장비가 틀린 값을 내면 모델이 그 값을 학습하고, 틀린 모델이 다음 실험을 고르고, 그 결과가 다시 모델에 들어간다. 사람이 매 단계를 보던 때는 자연히 걸러지던 오류가 고리 안에서 증폭된다. 이 장은 closed-loop의 구성 요소를 시뮬레이터 위에서 직접 만들고, batch 선택·안전 제약·측정 실패를 넣었을 때 무엇이 깨지는지, 그리고 모든 결정을 재현할 수 있게 기록하는 법을 다룬다.

## 학습 목표

- 모델·실험 선택·실험 수행이 이어지는 closed-loop 구조를 설계한다
- 자율 실험의 실패 양상을 안다
- closed-loop의 모든 결정을 재현할 수 있게 기록한다

## closed-loop의 구성

closed-loop 한 바퀴는 다섯 단계다.

1. **가설·모델**: 지금까지의 측정으로 대리모델을 만든다. 보통 GP(13.7장)나 앙상블처럼 불확실성을 내는 모델이다.
2. **실험 선택**: 획득함수(10.7장, 25.3장)로 다음 실험 조건을 고른다. 제약(안전·비용)과 batch 크기가 여기서 들어간다.
3. **실행**: 로봇·자동화 장비·시뮬레이터가 그 조건으로 실험을 수행한다.
4. **측정**: 결과를 측정하고, 측정이 정상인지 확인한다(장비 상태, 값의 범위, 반복 측정의 일치).
5. **모델 갱신**: 정상인 측정만 모델에 넣고, 모든 결정과 측정을 기록한다.

각 단계는 실패할 수 있고, 한 단계의 실패는 다음 바퀴의 입력이 된다. 그래서 closed-loop의 설계는 획득함수만큼이나 **단계 사이의 점검과 기록**에 달려 있다.

이 장에서는 시뮬레이터를 "가상 실험 장비"로 쓴다. 반응 온도(20–120 °C)와 반응 시간(5–60 min)을 정하면 생성물의 수율(0–1)을 측정 잡음과 함께 돌려주는 장비다. 수율 곡면에는 봉우리가 두 개 있고, 후보 조건은 21 × 21 격자다. 대리모델은 RBF 커널 GP다.

```python {exec="browser" session="loop"}
import numpy as np
from scipy.stats import norm

def true_yield(u):                                    # u: (..., 2) 정규화한 (온도, 시간) -> 참 수율 0–1
    a = 0.8 * np.exp(-((u[..., 0] - 0.65) ** 2 / 0.05 + (u[..., 1] - 0.4) ** 2 / 0.08))
    b = 0.55 * np.exp(-((u[..., 0] - 0.2) ** 2 + (u[..., 1] - 0.8) ** 2) / 0.02)
    return a + b

def units(u):                                         # 정규화 좌표 -> (온도 °C, 시간 min)
    return 20 + 100 * u[0], 5 + 55 * u[1]

g = np.linspace(0, 1, 21)
cand = np.array([(a, b) for a in g for b in g])       # (441, 2) 후보 실험 조건
NOISE = 0.02                                          # 측정 잡음 표준편차

def gp(X, y, Xs, ell=0.15, amp=0.3):                  # RBF 커널 GP: 사후 평균·표준편차
    k = lambda A, B: amp**2 * np.exp(-0.5 * ((A[:, None] - B[None]) ** 2).sum(-1) / ell**2)
    L = np.linalg.cholesky(k(X, X) + NOISE**2 * np.eye(len(X)))   # (n, n)
    v = np.linalg.solve(L, k(X, Xs))                  # (n, C)
    m = y.mean()
    return m + v.T @ np.linalg.solve(L, y - m), np.sqrt(np.clip(amp**2 - (v**2).sum(0), 1e-12, None))

best_j = np.argmax(true_yield(cand))
print("참 최적 조건: 온도 {:.0f} °C, 시간 {:.1f} min, 수율 {:.3f}".format(*units(cand[best_j]), true_yield(cand[best_j])))
print("두 번째 봉우리 수율: {:.3f}".format(true_yield(np.array([0.2, 0.8]))))
```

```output
참 최적 조건: 온도 85 °C, 시간 27.0 min, 수율 0.800
두 번째 봉우리 수율: 0.552
```

다음은 루프 자체다. 처음 3개 실험은 무작위로 고르고, 이후에는 UCB(upper confidence bound) $\mu(\mathbf{x}) + \beta\,\sigma(\mathbf{x})$, $\beta = 2$로 고른다. 실험마다 **무엇을 왜 골랐는지**(초기 무작위인지, 모델의 예측 평균과 표준편차가 얼마였는지)와 측정값을 기록에 남긴다. 그리고 같은 시드로 루프를 다시 돌려 기록이 똑같이 나오는지 확인한다.

```python {exec="browser" session="loop"}
def closed_loop(seed, n_exp=20, beta=2.0, n_init=3):
    rng = np.random.default_rng(seed)
    X, y, log = [], [], []
    for i in range(n_exp):
        if i < n_init:                                # 초기 실험은 무작위
            j, why = int(rng.integers(len(cand))), "초기 무작위"
        else:
            mu, sd = gp(np.array(X), np.array(y), cand)
            j = int(np.argmax(mu + beta * sd))        # UCB 획득함수
            why = f"UCB: 예측 {mu[j]:.3f} ± {sd[j]:.3f}"
        y_obs = float(true_yield(cand[j]) + NOISE * rng.normal())   # 가상 장비에서 측정
        X.append(cand[j]); y.append(y_obs)
        log.append({"실험": i, "온도_c": round(float(units(cand[j])[0]), 1), "시간_min": round(float(units(cand[j])[1]), 2),
                    "근거": why, "측정": round(y_obs, 3)})
    return log

log = closed_loop(seed=1)
print("같은 시드로 다시 돌린 결정 기록이 같다:", log == closed_loop(seed=1))
for r in log[2:5] + log[-1:]:
    print(r)
best = max(log, key=lambda r: r["측정"])
print(f"최고 측정: 실험 {best['실험']}, {best['온도_c']} °C, {best['시간_min']} min, 수율 {best['측정']}")
```

```output
같은 시드로 다시 돌린 결정 기록이 같다: True
{'실험': 2, '온도_c': 105.0, '시간_min': 18.75, '근거': '초기 무작위', '측정': 0.289}
{'실험': 3, '온도_c': 100.0, '시간_min': 32.5, '근거': 'UCB: 예측 0.182 ± 0.291', '측정': 0.459}
{'실험': 4, '온도_c': 120.0, '시간_min': 35.25, '근거': 'UCB: 예측 0.317 ± 0.276', '측정': 0.041}
{'실험': 19, '온도_c': 85.0, '시간_min': 24.25, '근거': 'UCB: 예측 0.778 ± 0.024', '측정': 0.77}
최고 측정: 실험 5, 85.0 °C, 27.0 min, 수율 0.812
```

기록을 읽으면 루프의 거동이 보인다. 실험 3, 4에서는 예측 표준편차(0.29, 0.28)가 예측 평균만큼 커서 탐색이 주도하고, 실험 19에서는 표준편차가 0.024로 줄어 이미 찾은 최적점 근처를 다시 재고 있다. 최고 측정은 실험 5의 0.812다. 참 최적 수율(0.800)보다 큰 값인데, 측정 잡음이 위로 튄 것이다. 여러 번 잰 값 중 최댓값은 위쪽으로 치우치므로, 최종 보고는 최고 측정값 하나가 아니라 대리모델의 예측 평균이나 반복 측정의 평균으로 한다.

같은 시드로 다시 돌린 기록이 같다는 것(`True`)은 당연해 보이지만 실제 시스템에서는 쉽게 깨진다. 획득함수 최적화에 무작위 재시작을 쓰거나, 병렬 실행의 완료 순서가 매번 다르거나, 모델 라이브러리 버전이 바뀌면 같은 데이터에서 다른 실험이 선택된다. 재현 가능한 기록에는 최소한 다음이 있어야 한다. 매 결정 시점의 입력 데이터(또는 그 해시), 모델과 획득함수의 설정과 시드, 선택된 조건과 그 근거(예측값·획득함수 값), 장비가 실제로 수행한 조건(설정값과 다를 수 있다), 측정값과 장비 상태, 사람의 개입 여부. 22.1장의 실험 추적 도구가 이 기록에도 쓰인다.

::widget{name="gp-regression"}

## 자율 실험실

**자율 실험실**(self-driving lab, SDL)은 위 고리의 "실행"과 "측정"을 로봇과 자동화 장비가 맡는 실험실이다. 액체 분주 로봇, 연속 흐름 반응기, 자동 박막 증착 장비, 자동 분광·회절 측정이 대표적인 구성 요소다. Abolhasani와 Kumacheva(2023)는 화학·재료 분야의 자율 실험실을 개관하며, 하드웨어 자동화와 데이터 기반 실험 선택을 결합한 시스템들을 정리했다.

구체적인 예로, Burger 외(2020)는 일반 실험실 안을 돌아다니며 기존 장비를 조작하는 이동형 로봇 화학자를 보고했다. 이 로봇은 광촉매 조성(변수 10개)을 batch 베이지안 탐색으로 고르며 8일 동안 688번의 실험을 자율적으로 수행했고, 초기 조성보다 활성이 약 6배 높은 혼합물을 찾았다. 이런 보고를 읽을 때는 무엇이 자동화되었고 무엇이 사람 몫이었는지(시약 보충, 장비 고장 대응, 탐색 공간 정의), 같은 예산의 대조군과 비교했는지를 함께 본다.

**자동화 수준.** 자율 실험은 전부 아니면 전무가 아니다. 대략 다음 단계로 나눌 수 있다.

| 수준 | 실험 선택 | 실험 수행 | 사람의 역할 |
|---|---|---|---|
| 1 | 사람 | 자동 장비(사람이 시작) | 모든 결정 |
| 2 | 알고리즘이 추천 | 사람 또는 자동 장비 | 추천을 승인·수정 |
| 3 | 알고리즘 | 자동 장비, 루프 자동 반복 | 목표·탐색 공간·제약 정의, 이상 대응 |
| 4 | 알고리즘이 가설·탐색 공간까지 갱신 | 자동 장비 | 감독 |

대부분의 실제 시스템은 2–3 수준이고, 이 장의 시뮬레이션은 3 수준이다. 수준이 높을수록 사람의 판단이 들어가던 점검을 코드로 옮겨야 한다. 아래 "운영 문제"가 그 목록이다.

:::note[분야 예시]
재료 분야에서는 박막 조성이나 열처리 조건을 바꿔 가며 전도도·광학 특성을 측정하는 루프, 화학에서는 연속 흐름 반응기에서 온도·유량·농도를 바꿔 수율을 높이는 루프가 흔하다. 생명과학에서는 액체 분주 로봇으로 배양 조건이나 단백질 변이체를 시험하는 루프가 있다. 배터리 연구(30.6장)에서는 충전 프로토콜을 고르고 셀을 충방전해 수명을 재는 루프가 같은 구조다. 측정 한 번이 몇 주씩 걸리므로 batch 선택과 조기 예측 모델이 함께 쓰인다. 계산 분야에서도 같은 구조가 쓰인다. 고비용 시뮬레이션(DFT 계산, 유체 시뮬레이션)을 "장비"로 두면 실패한 계산(수렴 실패)을 어떻게 다루느냐가 측정 실패와 같은 문제가 된다.
:::

## 다목적과 제약

**batch 선택.** 실제 장비는 여러 실험을 동시에 돌리는 경우가 많다. 반응기가 4대면 한 라운드에 조건 4개를 정해야 한다. 획득함수 점수 상위 4개를 그대로 고르면 문제가 생긴다. 점수가 높은 곳은 서로 붙어 있기 마련이라, 거의 같은 조건을 네 번 재게 된다. 간단한 해결책이 **kriging believer**다(Ginsbourger 외, 2010). 하나를 고른 뒤 그 점의 측정값이 대리모델의 예측 평균과 같다고 **가정**해 모델을 갱신한다. 그러면 그 점 근처의 불확실성이 줄어들어 다음 선택은 다른 곳으로 간다. 이 과정을 batch 크기만큼 반복한다.

```python {exec="browser" session="loop"}
def pick_batch(X, y, q, beta=2.0, believer=True):
    X, y = list(X), list(y)
    if not believer:                                  # 단순 상위 q개: 한 번 계산한 UCB 점수의 상위 q
        mu, sd = gp(np.array(X), np.array(y), cand)
        return [int(j) for j in np.argsort(-(mu + beta * sd))[:q]]
    batch = []
    for _ in range(q):
        mu, sd = gp(np.array(X), np.array(y), cand)
        j = int(np.argmax(mu + beta * sd))
        batch.append(j)
        X.append(cand[j]); y.append(mu[j])            # kriging believer: 예측 평균을 측정값인 척 넣는다
    return batch

def run_batch(seed, q, rounds, believer=True):
    rng = np.random.default_rng(seed)
    idx = [int(j) for j in rng.integers(len(cand), size=3)]          # 초기 3개는 무작위
    y = [float(true_yield(cand[j]) + NOISE * rng.normal()) for j in idx]
    spread = []
    for _ in range(rounds):
        b = pick_batch(cand[idx], y, q, believer=believer)
        if q > 1:                                     # batch 안에서 가장 가까운 두 조건의 거리
            spread.append(min(np.linalg.norm(cand[i] - cand[j]) for i in b for j in b if i < j))
        for j in b:
            idx.append(j); y.append(float(true_yield(cand[j]) + NOISE * rng.normal()))
    best = np.maximum.accumulate(true_yield(cand[idx]))             # 지금까지 고른 조건의 참 수율 최고
    regret = true_yield(cand).max() - best[[3 + 4 - 1, 3 + 8 - 1, 3 + 16 - 1]]   # 실험 4·8·16개 뒤
    return list(regret) + [np.mean(spread) if spread else np.nan]

for name, q, rounds, bel in [("순차 1개씩", 1, 16, True), ("4개 batch, believer", 4, 4, True),
                             ("4개 batch, 단순 상위 4", 4, 4, False)]:
    R = np.array([run_batch(s, q, rounds, bel) for s in range(20)])    # (시드, 4)
    gap = f"{np.nanmean(R[:, 3]):.3f}" if q > 1 else "  -  "
    print(f"{name:18s} 라운드 {rounds:2d}회  평균 regret  실험 4개 {R[:, 0].mean():.3f}  8개 {R[:, 1].mean():.3f}  16개 {R[:, 2].mean():.3f}  batch 최소 간격 {gap}")
```

```output
순차 1개씩             라운드 16회  평균 regret  실험 4개 0.090  8개 0.027  16개 0.000  batch 최소 간격   -  
4개 batch, believer 라운드  4회  평균 regret  실험 4개 0.146  8개 0.058  16개 0.004  batch 최소 간격 0.231
4개 batch, 단순 상위 4  라운드  4회  평균 regret  실험 4개 0.216  8개 0.053  16개 0.036  batch 최소 간격 0.052
```

실험 수로 보면 순차 선택이 가장 효율적이다. 매 실험이 앞선 실험의 결과를 모두 보기 때문이다. kriging believer batch는 실험 4개, 8개 시점에는 순차보다 regret이 크지만 실험 16개에서 거의 같아진다(0.004). 그리고 **라운드 수는 4분의 1**이다. 라운드 하나가 하루 걸리는 실험이면 16일이 4일이 된다. 점수 상위 4개를 그대로 고른 batch는 batch 안의 가장 가까운 두 조건 사이 거리가 평균 0.052(격자 한 칸 크기 0.05)로, 사실상 이웃한 조건을 한꺼번에 잰다. 실험 16개 뒤 regret(0.036)도 가장 나쁘다. batch 선택에서는 **다양성**이 필요하다는 25.3장의 말을 숫자로 확인한 것이다.

**안전 제약.** 어떤 조건은 위험하거나 장비를 망가뜨린다. 반응기 압력이 한계를 넘는 조건, 너무 높은 전압, 독성 부산물이 생기는 조합이 그렇다. 제약이 미리 알려진 식이면 후보에서 빼면 된다. 그러나 흔히 제약도 실험해 봐야 안다. 이 경우 제약을 **두 번째 대리모델**로 학습하고, 획득함수에 "제약을 만족할 확률"을 곱한다(Gelbart 외, 2014).

$$
\alpha_{\text{c}}(\mathbf{x}) = \mathrm{EI}(\mathbf{x}) \cdot \Pr\big[c(\mathbf{x}) < c_{\max}\big] = \mathrm{EI}(\mathbf{x}) \cdot \Phi\!\left(\frac{c_{\max} - \mu_c(\mathbf{x})}{\sigma_c(\mathbf{x})}\right).
$$

$c(\mathbf{x})$는 제약 변수(여기서는 압력), $\mu_c, \sigma_c$는 제약 GP의 예측 평균과 표준편차다. 가상 장비에 압력을 함께 기록하게 하고, 압력이 4 bar를 넘으면 장비가 실험을 중단한다고 하자. 중단된 실험은 수율을 얻지 못하므로 수율 0으로 기록한다. 이 제약 때문에 원래의 최적점(수율 0.800)은 쓸 수 없고, 한계 안의 최고 수율은 0.672다.

```python {exec="browser" session="loop"}
P_MAX = 4.0                                           # 반응기 압력 한계 [bar]
def pressure_bar(u):                                  # 장비가 함께 기록하는 압력(실험 전에는 모른다)
    return 1 + 8 * u[..., 0] * (0.3 + u[..., 1])

feasible = pressure_bar(cand) < P_MAX
print(f"제약 없는 최고 수율 {true_yield(cand).max():.3f}, 압력 한계 안 최고 수율 {true_yield(cand[feasible]).max():.3f}")

def constrained_loop(seed, use_constraint, n_exp=20):
    rng = np.random.default_rng(seed)
    idx = [int(j) for j in rng.choice(np.flatnonzero(feasible), 3)]  # 초기 3개는 안전한 조건에서
    y = [float(true_yield(cand[j]) + NOISE * rng.normal()) for j in idx]
    p = [float(pressure_bar(cand[j])) for j in idx]
    while len(idx) < n_exp:
        mu, sd = gp(cand[idx], np.array(y), cand)
        z = (mu - max(y)) / sd
        score = (mu - max(y)) * norm.cdf(z) + sd * norm.pdf(z)       # EI (최대화)
        if use_constraint:                                           # 압력 GP로 한계 안일 확률을 곱한다
            mp, sp = gp(cand[idx], np.array(p), cand, ell=0.4, amp=2.0)
            score = score * norm.cdf((P_MAX - mp) / sp)
        j = int(np.argmax(score))
        safe = pressure_bar(cand[j]) < P_MAX
        idx.append(j); p.append(float(pressure_bar(cand[j])))
        y.append(float(true_yield(cand[j]) + NOISE * rng.normal()) if safe else 0.0)   # 중단된 실험은 수율 0으로 기록
    ok = [j for j in idx if feasible[j]]
    return int((~feasible[idx]).sum()), true_yield(cand[feasible]).max() - true_yield(cand[ok]).max()

for use in (False, True):
    R = np.array([constrained_loop(s, use) for s in range(20)])     # (시드, 2)
    print(f"제약 모델 {'사용' if use else '없음'}: 한계 위반 실험 평균 {R[:, 0].mean():.1f}/20회, 안전 영역 regret 평균 {R[:, 1].mean():.3f}")
```

```output
제약 없는 최고 수율 0.800, 압력 한계 안 최고 수율 0.672
제약 모델 없음: 한계 위반 실험 평균 6.0/20회, 안전 영역 regret 평균 0.123
제약 모델 사용: 한계 위반 실험 평균 2.4/20회, 안전 영역 regret 평균 0.030
```

제약 모델 없이 EI만 쓰면 실험 20번 중 평균 6번이 압력 한계를 넘는다. 제약 모델을 쓰면 2.4번으로 줄고, 안전 영역 안에서 찾은 최고 수율도 참값에 훨씬 가깝다(regret 0.123 → 0.030). 위반이 0이 되지 않는 것은 제약 GP도 처음에는 모르기 때문이다. 실제 장비에서 한 번의 위반도 허용할 수 없다면 확률 대신 보수적인 문턱(예: $\Pr[c < c_{\max}] > 0.99$인 후보만 허용)을 쓰고, 초기 실험은 확실히 안전한 영역에서 시작한다. 제약 모델이 없을 때 regret도 나쁜 데는 이유가 있다. 중단된 실험을 수율 0으로 넣으면 대리모델은 한계 근처에서 수율이 갑자기 떨어진다고 배우고, 그 왜곡이 한계 안쪽의 예측까지 흔든다. 측정할 수 없었던 값을 지어낸 값으로 채우면 안 된다는, 다음 절 실패 사례의 한 형태다.

::quiz{id="constraint-probability"}

**다목적.** 수율만 중요한 경우는 드물다. 수율과 순도, 성능과 비용, 효율과 안정성처럼 여러 목적이 충돌한다. 10.7장에서 본 것처럼 목적들의 가중합으로 하나로 만들 수도 있지만, 가중치를 미리 정하기 어렵고 front가 볼록이 아닌 부분의 해는 어떤 가중치로도 찾지 못한다. 그래서 자율 실험에서는 측정된 점들의 **Pareto front**(모든 목적에서 같거나 낫고 하나 이상에서 더 나은 다른 점이 없는 점들)를 유지하고, 그 front를 가장 많이 넓힐 실험을 고르는 다목적 획득함수(기대 hypervolume 개선 등)를 쓰기도 한다. 어느 경우든 루프가 내는 결과는 "최적 조건 하나"가 아니라 trade-off 곡선이고, 그중 무엇을 고를지는 사람이 정한다. 연습문제 `pareto-front`에서 측정된 점들의 Pareto front를 직접 구한다.

::widget{name="pareto-front"}

## 운영 문제

알고리즘이 맞아도 장비와 데이터가 틀리면 루프는 틀린 곳으로 간다. 운영에서 자주 만나는 문제는 다음과 같다.

**측정 실패.** 시료 손실, 장비 오류, 통신 끊김으로 값이 없거나 틀린 값이 나온다. 더 위험한 것은 값이 **그럴듯하게** 나오는 실패다. 검출기가 포화되면 최댓값이 나오고, 기준선이 틀어지면 모든 값이 일정하게 밀린다. 가상 장비에 두 가지 실패를 넣는다. 실험의 15%는 값을 잃고(실패), 고온·단시간 영역에서는 검출기가 포화되어 언제나 1.0을 낸다(포화). 장비는 매 측정에 상태(`정상`, `실패`, `포화`)를 함께 보고한다. 상태를 확인하지 않는 루프는 실패를 0으로, 포화값을 그대로 학습한다. 상태를 확인하는 루프는 정상이 아닌 측정을 학습에서 빼고, 포화가 확인된 조건은 다시 고르지 않는다. 두 루프 모두 실패한 실험도 예산 20번에 포함된다.

```python {exec="browser" session="loop"}
def instrument(j, rng):                               # 가상 장비: (측정값, 상태)
    if rng.random() < 0.15:
        return np.nan, "실패"                          # 시료 손실 등으로 값이 없다
    if cand[j, 0] > 0.85 and cand[j, 1] < 0.2:        # 고온·단시간 영역에서 검출기가 포화된다
        return 1.0, "포화"
    return float(true_yield(cand[j]) + NOISE * rng.normal()), "정상"

def robust_loop(seed, check_status, n_exp=20, beta=2.0):
    rng = np.random.default_rng(seed)
    idx, y, banned = [], [], set()
    for _ in range(n_exp):                            # 실패한 실험도 예산을 쓴다
        if len(idx) < 3:                              # 쓸 수 있는 측정이 3개가 될 때까지 무작위
            j = int(rng.integers(len(cand)))
        else:
            mu, sd = gp(cand[idx], np.array(y), cand)
            score = mu + beta * sd
            score[list(banned)] = -np.inf             # 포화가 확인된 조건은 다시 고르지 않는다
            j = int(np.argmax(score))
        val, status = instrument(j, rng)
        if check_status and status != "정상":         # 상태를 확인하면: 학습에서 빼고 기록만 남긴다
            banned |= {j} if status == "포화" else set()
            continue
        idx.append(j); y.append(0.0 if np.isnan(val) else val)       # 확인하지 않으면: 실패는 0, 포화값은 그대로
    k = int(np.argmax(y))                             # 루프가 보고하는 최고 조건
    return y[k], true_yield(cand[idx[k]]), sum(cand[j, 0] > 0.85 and cand[j, 1] < 0.2 for j in idx)

for check in (False, True):
    R = np.array([robust_loop(s, check) for s in range(20)])        # (시드, 3)
    print(f"상태 확인 {'함' if check else '안 함'}: 보고한 최고 수율 평균 {R[:, 0].mean():.3f}, "
          f"그 조건의 참 수율 평균 {R[:, 1].mean():.3f}, 학습에 들어간 포화 영역 실험 평균 {R[:, 2].mean():.1f}회")
```

```output
상태 확인 안 함: 보고한 최고 수율 평균 0.887, 그 조건의 참 수율 평균 0.299, 학습에 들어간 포화 영역 실험 평균 4.2회
상태 확인 함: 보고한 최고 수율 평균 0.794, 그 조건의 참 수율 평균 0.784, 학습에 들어간 포화 영역 실험 평균 0.0회
```

상태를 확인하지 않은 루프는 평균 0.887의 "최고 수율"을 보고하지만, 그 조건의 참 수율은 평균 0.299다. 포화된 측정 1.0이 대리모델에 "고온·단시간에 높은 봉우리가 있다"고 가르쳤고, UCB가 그 봉우리를 더 확인하러 가면서 포화 영역 실험이 평균 4.2번 학습에 들어갔다. 틀린 측정이 다음 실험을 끌어당기고, 그 실험이 다시 틀린 측정을 낸 것이다. 상태를 확인한 루프는 같은 예산에서 참 수율 0.784인 조건을 보고한다. 보고값(0.794)과 참값의 차이도 측정 잡음 수준이다.

::quiz{id="saturation-feedback"}

**장비 드리프트.** 센서 감도, 촉매 활성, 광원 세기는 시간에 따라 천천히 변한다. 드리프트는 실험 순서와 구별할 수 없다. 루프가 시간이 지나며 최적점 근처로 모이므로, 드리프트가 수율을 낮추면 "최적점 근처가 생각보다 나쁘다"는 잘못된 결론이 나온다. 대응은 **기준 시료를 주기적으로 재는 것**이다. 같은 조건(대조 조건)을 정해진 간격으로 반복 측정하고, 그 값이 변하면 루프를 멈추고 보정한다. 대조 측정은 예산을 쓰지만, 그것 없이 나온 결과는 해석할 수 없다.

**사람의 개입 지점.** 수준 3의 루프에서도 사람이 반드시 결정해야 하는 지점이 있다.

- **시작 전**: 목적함수, 탐색 공간, 안전 제약, 예산, 멈춤 조건. 루프는 이 정의 안에서만 최적화한다. 정의가 틀리면 루프는 틀린 문제를 효율적으로 푼다.
- **경보가 울릴 때**: 장비 상태 이상, 대조 측정의 변화, 예측과 크게 다른 측정(예: 예측 평균에서 표준편차의 4배 이상 벗어남), 같은 조건의 반복 실패. 이때 루프는 자동으로 멈추고 사람의 확인을 기다리는 것이 기본값이어야 한다.
- **결론을 낼 때**: 루프가 찾은 최적 조건을 독립적인 반복 실험으로 확인하는 것은 사람의 몫이다. 루프 안의 측정은 루프가 고른 조건에 치우쳐 있다.
- **개입 자체의 기록**: 사람이 루프를 멈추거나 조건을 바꾸면 그 사실과 이유도 기록에 남긴다. 그렇지 않으면 기록만으로 루프를 재현할 수 없다.

**기록과 재현.** 셀 2의 기록처럼 매 결정의 입력·설정·근거·결과를 남긴다. 실제 장비에서는 같은 기록으로 루프를 "다시 돌려" 볼 수는 없지만, 기록된 측정값을 시뮬레이터 대신 넣고 결정 부분만 다시 실행해 같은 실험이 선택되는지 확인할 수 있다(replay). 이것이 closed-loop 결과의 재현성 점검이다. 22.2장의 재현성 원칙(환경·버전·시드 고정)이 그대로 적용된다.

## 실습

시뮬레이터를 "가상 실험 장비"로 두고 closed-loop 최적화를 구성해, 측정 잡음과 실패를 넣었을 때의 거동을 관찰한다. 코드는 `labs/scientific-discovery/autonomous-experimentation/cells.py`에 있다.

1. 셀 2의 기록을 JSON Lines 파일로 저장하고, 저장된 측정값만으로 결정 부분을 다시 실행해 같은 조건이 선택되는지 확인하는 replay 함수를 만든다.
2. 측정 잡음을 0.02에서 0.05, 0.1로 키우며 최종 보고값(최고 측정, 대리모델 예측 최댓값, 최고 조건의 3회 반복 평균)이 참 수율에서 얼마나 벗어나는지 비교한다.
3. 가상 장비에 시간에 따라 감도가 1%씩 떨어지는 드리프트를 넣는다. 다섯 실험마다 대조 조건을 재서 드리프트를 감지하고 보정하는 루프와, 그렇지 않은 루프의 결과를 비교한다.
4. 셀 5의 루프에 "예측 평균에서 표준편차의 4배 이상 벗어난 측정은 한 번 더 잰다"는 규칙을 넣고, 장비 상태 정보 없이도 포화를 걸러낼 수 있는지 확인한다.

연습문제 두 개에서 획득함수와 다목적 Pareto front를 직접 구현한다.

::exercise{id="acquisition-functions"}

::exercise{id="pareto-front"}

## 실패 사례

- **측정 실패를 정상 데이터로 학습하는 것.** 실패를 0으로, 포화값을 그대로 넣으면 틀린 측정이 다음 실험을 그쪽으로 끌어당기고 오류가 증폭된다. 셀 5에서 보고한 최고 수율 0.887의 실제 수율은 0.299였다. 장비 상태를 매 측정과 함께 받고, 정상이 아닌 측정은 학습에서 빼되 기록에는 남긴다.
- **루프의 결정 기록을 남기지 않는 것.** 무엇을 왜 골랐는지 없으면 결과를 재현할 수도, 잘못된 결정이 어디서 시작되었는지 찾을 수도 없다.
- **최고 측정값을 그대로 보고하는 것.** 잡음 있는 측정 여러 개의 최댓값은 위로 치우친다. 셀 2의 최고 측정 0.812는 참 최적값 0.800보다 컸다. 최종 조건은 독립적인 반복 측정으로 확인한다.
- **batch를 점수 상위 몇 개로 채우는 것.** 이웃한 조건을 한꺼번에 재 예산을 낭비한다.
- **알 수 없는 제약을 무시하거나, 중단된 실험을 임의의 값으로 채우는 것.** 위반이 늘고, 채운 값이 대리모델을 왜곡한다.

## 통과 기준

- 시뮬레이터 대상 closed-loop 최적화를 구성하고, 모든 결정(입력 데이터, 설정과 시드, 선택 근거, 측정값과 장비 상태)을 재현할 수 있는 기록과 함께 결과를 보고한다.
- 측정 실패·이상값·제약을 넣은 조건에서 루프의 거동을 같은 예산의 기준 루프와 비교해 보인다.
- 연습문제 두 개(`acquisition-functions`, `pareto-front`)의 테스트를 통과한다.

## 참고 문헌

- Jones, Schonlau, Welch (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455–492. — expected improvement와 EGO
- Ginsbourger, Le Riche, Carraro (2010). Kriging is well-suited to parallelize optimization. In *Computational Intelligence in Expensive Optimization Problems*, Springer, 131–162. — 다점 EI와 kriging believer 같은 batch 선택 전략
- Gelbart, Snoek, Adams (2014). Bayesian optimization with unknown constraints. *UAI 2014*. arXiv:1403.5607. — 실험해야 알 수 있는 제약을 대리모델로 다루는 베이지안 최적화
- Burger 외 (2020). A mobile robotic chemist. *Nature*, 583, 237–241. — 8일간 688회 실험을 자율 수행한 이동형 로봇 화학자
- Abolhasani, Kumacheva (2023). The rise of self-driving labs in chemical and materials sciences. *Nature Synthesis*, 2, 483–492. — 화학·재료 분야 자율 실험실 리뷰
