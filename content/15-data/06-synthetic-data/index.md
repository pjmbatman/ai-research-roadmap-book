---
id: data/synthetic-data
title: "합성 데이터와 시뮬레이션 데이터"
title_en: "Synthetic and Simulated Data"
tier: advanced
status: draft
prerequisites: [data/dataset-versioning, numerical-methods/ode-solvers]
---

# 합성 데이터와 시뮬레이션 데이터

실험은 비싸고 느리다. 배터리 하나의 수명을 재는 데 몇 달이 걸리고, 극한 조건의 실험은 위험하거나 불가능하다. 그래서 과학 머신러닝에서는 시뮬레이터로 데이터를 만들어 학습에 쓰는 일이 흔하다. 시뮬레이션 데이터는 무한히 만들 수 있고, label이 정확하며(시뮬레이터가 정한 파라미터가 곧 정답이다), 위험한 조건도 탐색할 수 있다.

대가도 있다. 모델은 실제 세계가 아니라 **시뮬레이터**를 배운다. 시뮬레이터가 모르는 물리, 시뮬레이터에 없는 센서의 특성은 학습 데이터에 없다. 이 장은 시뮬레이션 데이터셋을 재현 가능하게 설계하고, 실험 설계로 파라미터 공간을 효율적으로 덮고, 시뮬레이션과 실제 사이의 간극(sim-to-real gap)을 측정하고 줄이는 방법을 다룬다. 9.6장의 ODE solver로 만든 감쇠 진동자를 예로 쓴다.

## 학습 목표

- 시뮬레이션 데이터셋을 설계한다
- 실험 설계로 입력 공간을 덮는다
- sim-to-real 간극을 평가한다

## 시뮬레이션 데이터의 생성

**왜 필요한가.** 시뮬레이션 데이터는 실측 데이터보다 다시 만들기 쉬워 보인다. 코드가 있으니 다시 돌리면 된다. 그러나 실제로는 어떤 파라미터 범위로, 어떤 표본으로, 어떤 solver 설정으로, 어떤 시드로 만들었는지 기록하지 않아 같은 데이터를 다시 만들지 못하는 일이 많다. 시뮬레이션 데이터셋은 **설정의 함수**다. 설정을 기록하면 데이터를 저장하지 않아도 되고, 기록하지 않으면 데이터를 저장해도 그 의미를 알 수 없다.

**구성 요소.** 시뮬레이션 데이터셋 하나는 다음으로 정해진다.

| 요소 | 예 | 기록 방법 |
|---|---|---|
| 시뮬레이터 | 모델 방정식, 코드 | 코드 커밋, 시뮬레이터 버전 |
| 파라미터 | 감쇠비 $\zeta$, 고유진동수 $\omega$의 범위 | 설정 파일의 `bounds` |
| 초기조건 | $x(0) = 1$, $v(0) = 0$ | 설정 파일 |
| 입력 프로파일 | 외력, 충방전 전류 프로파일 | 프로파일 파일과 해시 |
| 실험 설계 | grid, LHS, Sobol, 표본 수 | 설정 파일의 `design`, `n` |
| 난수 시드 | 설계·잡음의 시드 | 설정 파일의 `seed` |
| solver 설정 | step 크기, tolerance, 적분 방법 | 설정 파일. 다르면 출력이 달라진다(9.6장) |

**예.** 비선형 강성 항이 있는 감쇠 진동자(Duffing 방정식)

$$
\ddot{x} + 2\zeta\omega\,\dot{x} + \omega^2 x + \alpha x^3 = 0, \qquad x(0) = 1,\ \dot{x}(0) = 0
$$

을 시뮬레이터로 쓴다. $\zeta$는 감쇠비, $\omega$는 고유진동수(rad/s), $\alpha$는 비선형 강성의 크기다. $\alpha = 0$이면 선형 감쇠 진동자다. 고전 4차 Runge–Kutta(RK4)로 10 s 동안 0.01 s 간격으로 적분하고 0.5 s마다 위치를 기록해, 샘플마다 길이 20의 궤적을 얻는다. 과제는 궤적에서 감쇠비 $\zeta$를 추정하는 것이다. 설정 전체를 하나의 딕셔너리로 모으고, 그 해시를 데이터셋의 이름표로 쓴다(15.5장).

```python {exec="browser" session="synth"}
import hashlib
import json
import numpy as np
from scipy.stats import qmc

def simulate(zeta, omega_rad_s, alpha=0.0, t_end_s=10.0, dt_s=0.01, every=50):
    """x'' + 2ζω x' + ω² x + α x³ = 0, x(0) = 1, v(0) = 0을 RK4로 푼다. 반환 (N, T)."""
    z, w = np.asarray(zeta, float), np.asarray(omega_rad_s, float)       # (N,), (N,)
    x, v, out = np.ones_like(z), np.zeros_like(z), []
    f = lambda x, v: (v, -2 * z * w * v - w**2 * x - alpha * x**3)
    for k in range(int(round(t_end_s / dt_s))):
        if k % every == 0:
            out.append(x)                                                # 0.5 s마다 기록
        a1, b1 = f(x, v)
        a2, b2 = f(x + 0.5 * dt_s * a1, v + 0.5 * dt_s * b1)
        a3, b3 = f(x + 0.5 * dt_s * a2, v + 0.5 * dt_s * b2)
        a4, b4 = f(x + dt_s * a3, v + dt_s * b3)
        x, v = x + dt_s / 6 * (a1 + 2 * a2 + 2 * a3 + a4), v + dt_s / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
    return np.stack(out, axis=1)

config = {"simulator": "duffing-rk4", "version": 1, "design": "lhs", "n": 400, "seed": 0,
          "bounds": {"zeta": [0.05, 0.5], "omega_rad_s": [1.0, 4.0]}, "alpha": 0.0, "dt_s": 0.01}

def make_dataset(cfg):
    unit = qmc.LatinHypercube(d=2, seed=cfg["seed"]).random(cfg["n"])     # (n, 2) in [0, 1)
    lo, hi = zip(*cfg["bounds"].values())
    P = qmc.scale(unit, lo, hi)                                           # (n, 2): ζ, ω
    return P, simulate(P[:, 0], P[:, 1], cfg["alpha"], dt_s=cfg["dt_s"])

P, X_sim = make_dataset(config)                                           # (400, 2), (400, 20)
cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
print("설정 해시:", cfg_hash[:16], " 데이터 shape:", X_sim.shape)
print("같은 설정으로 다시 만든 데이터가 같은가:", np.array_equal(make_dataset(config)[1], X_sim))
```

```output
설정 해시: cb57d80de0bf4807  데이터 shape: (400, 20)
같은 설정으로 다시 만든 데이터가 같은가: True
```

설정 해시 하나가 이 데이터셋을 가리킨다. 같은 설정으로 다시 만들면 바이트 단위로 같은 데이터가 나온다. 이 확인은 같은 라이브러리 버전과 같은 플랫폼에서 성립한다. 부동소수점 연산의 순서가 다른 환경에서는 마지막 자리가 달라질 수 있으므로, 환경도 함께 기록한다(3.6장).

:::tip[연구 팁]
설정 파일을 먼저 쓰고, 생성 스크립트는 설정 파일만 받게 만든다(`python make_data.py configs/sim_v1.yaml`). 생성된 데이터 디렉터리에는 설정 파일 사본, 설정 해시, 코드 커밋, 출력의 manifest 해시를 함께 둔다. 그러면 데이터를 지워도 다시 만들 수 있고, 다시 만든 데이터가 같은지도 확인할 수 있다.
:::

::quiz{id="config-reproducibility"}

## 실험 설계

**왜 필요한가.** 시뮬레이션 한 번이 몇 분~몇 시간 걸리면 실행 수가 곧 예산이다. 같은 예산으로 파라미터 공간을 얼마나 잘 덮느냐가 데이터셋의 질을 정한다. 특정 영역이 비어 있으면 모델은 그 영역에서 외삽해야 한다.

**네 가지 설계.** $d$차원 단위 정육면체 $[0, 1)^d$에서 $n$개의 점을 고른 뒤 각 파라미터의 범위로 옮긴다.

- **grid**: 축마다 $k$단계를 고르고 모든 조합을 실행한다. 실행 수는 $k^d$로, 차원이 늘면 금방 불가능해진다. 또 축마다 $k$개의 값만 시험한다.
- **random**: 균등 난수로 $n$개를 뽑는다. 차원과 무관하게 $n$을 정할 수 있지만, 우연히 몰리거나 빈 곳이 생긴다.
- **Latin hypercube**(LHS): 각 축을 $n$개 구간으로 나누어, 모든 축에서 구간마다 정확히 한 점이 있게 한다(McKay 외, 1979). 축마다 $n$개의 서로 다른 값을 시험한다.
- **Sobol 수열**: 공간을 체계적으로 채우도록 설계된 결정적 저불일치(low-discrepancy) 수열이다. 점의 수가 2의 거듭제곱일 때 균형이 가장 좋다.

**커버리지 평가.** 두 가지를 본다. 축마다의 커버리지(1차원 투영이 몇 개 구간을 덮는가)와 **discrepancy**(점들의 경험 분포가 균등분포에서 얼마나 벗어나는가, 작을수록 고르다)다. 같은 16점으로 비교한다.

```python {exec="browser" session="synth"}
n = 16
g = (np.arange(4) + 0.5) / 4                                      # (4,) 축마다 4단계
designs = {
    "grid":   np.array([(a, b) for a in g for b in g]),           # (16, 2)
    "random": np.random.default_rng(0).random((n, 2)),
    "LHS":    qmc.LatinHypercube(d=2, seed=0).random(n),
    "Sobol":  qmc.Sobol(d=2, seed=0).random(n),
}
for name, S in designs.items():
    cover = len(np.unique(np.floor(S[:, 0] * n)))                 # ζ축 투영이 덮는 칸 수(16칸 중)
    print(f"{name:6s}  ζ축 16칸 중 {cover:2d}칸  discrepancy {qmc.discrepancy(S):.4f}")
```

```output
grid    ζ축 16칸 중  4칸  discrepancy 0.0114
random  ζ축 16칸 중  8칸  discrepancy 0.0395
LHS     ζ축 16칸 중 16칸  discrepancy 0.0025
Sobol   ζ축 16칸 중 16칸  discrepancy 0.0020
```

grid는 각 축에서 4개 값만 시험한다. 출력이 주로 $\zeta$에 의존한다면, 16번의 실행 중 $\zeta$에 대한 정보는 4가지 값에 대한 것뿐이다. random은 16칸 중 8칸만 덮었고 discrepancy가 가장 크다. LHS와 Sobol은 축마다 16칸을 모두 덮고 discrepancy도 가장 작다. 차원이 높을수록, 그리고 일부 파라미터만 중요할수록 이 차이가 커진다.

::widget{name="doe-coverage"}

::quiz{id="grid-projection"}

::quiz{id="grid-curse"}

**연구에서의 의미.** 실험 설계는 파라미터의 **범위**를 정하는 결정과 짝을 이룬다. 범위가 실제 계의 가능한 값을 덮지 못하면, 아무리 고르게 채워도 실제 데이터는 학습 범위 밖에 있다. 범위는 문헌·실측 데이터·물리적 제약으로 정하고 근거를 설정 파일 옆에 기록한다. 균일한 설계 대신 관심 영역(예: 고장 경계 근처)에 점을 더 두는 적응적 설계도 있다. 이 경우 학습 분포가 균일하지 않다는 점을 데이터 문서에 쓴다.

::quiz{id="lhs-property"}

:::note[분야 예시]
기후 모델의 파라미터 불확실성 연구는 수십 개의 물리 파라미터를 LHS로 뽑아 앙상블을 돌린다. 한 번의 실행이 매우 비싸 완전 격자는 불가능하기 때문이다. 재료 설계에서는 조성·공정 조건의 공간을 Sobol 수열로 덮어 계산을 배치한다. 배터리 전기화학 모델의 파라미터 추정에서도 확산 계수·반응 속도 같은 파라미터의 범위를 LHS로 덮어 시뮬레이션 데이터베이스를 만든다.
:::

## 합성 데이터의 위험

**시뮬레이터 가정의 학습.** 시뮬레이터로 만든 데이터에는 시뮬레이터의 모든 가정이 들어 있다. 모델은 실제 물리가 아니라 그 가정을 배운다. 시뮬레이터가 선형 강성을 가정하면, 모델은 비선형 강성이 있는 실제 계에서 체계적으로 틀린다. 시뮬레이터에 센서 잡음·offset·표본화 지터가 없으면, 모델은 깨끗한 신호에만 맞춰진다. 이 오류는 시뮬레이션 test에서 보이지 않는다. 학습과 test가 같은 가정을 공유하기 때문이다.

**실제 분포와의 차이.** 실험 설계는 파라미터 공간을 고르게 덮지만, 실제 계의 파라미터는 고르게 분포하지 않는다. 실제 셀은 대부분 정상 범위에 몰려 있고, 고장은 드물다. 시뮬레이션 데이터로 학습한 모델의 성능을 시뮬레이션 분포로 평가하면, 실제로 자주 만나는 영역과 드문 영역의 가중치가 다르게 매겨진다.

**생성 모델로 만든 합성 데이터.** 실측 데이터로 학습한 생성 모델(GAN, diffusion 모델 등)로 합성 데이터를 만드는 방법도 있다. 이 데이터는 학습에 쓴 실측 데이터의 분포 안에서만 그럴듯하고, 그 분포에 없는 현상을 만들지 못한다. 생성 모델 학습에 test 데이터가 들어가면 누수가 된다(15.4장). 개인정보 보호를 위한 합성 데이터도 원 데이터의 개인 기록을 거의 그대로 재현할 수 있으므로, 공개 전에 원 데이터와의 근접도를 점검한다(15.5장).

## sim-to-real

**왜 필요한가.** 시뮬레이션으로 학습한 모델을 실제 계에 쓰려면, 두 분포가 얼마나 다른지(domain gap)와 그 차이가 성능을 얼마나 떨어뜨리는지를 재야 한다. 앞의 감쇠 진동자 시뮬레이터는 $\alpha = 0$(선형 강성)을 가정한다. "실제 계"에는 시뮬레이터가 모르는 비선형 강성 $\alpha = 1$, 센서 offset 0.02, 잡음(표준편차 0.02)이 있다고 하자.

**간극 측정.** 두 가지를 잰다.

- **성능 간극**: 시뮬레이션 test와 실측 test에서 같은 모델의 오차를 비교한다. 실측 label이 조금이라도 있어야 한다.
- **분포 간극**: 시뮬레이션 샘플과 실측 샘플을 섞어 출처를 맞히는 분류기(domain classifier)를 학습한다. 두 분포가 같으면 어떤 분류기도 AUC 0.5를 넘지 못한다. AUC가 높을수록 간극이 크다. 이 방법은 실측 label이 없어도 쓸 수 있다.

```python {exec="browser" session="synth"}
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(1)
P_real = np.column_stack([rng.uniform(0.05, 0.5, 200), rng.uniform(1.0, 4.0, 200)])  # (200, 2)
X_real = (simulate(P_real[:, 0], P_real[:, 1], alpha=1.0)        # 실제 계: 비선형 강성 α = 1
          + 0.02 + rng.normal(0, 0.02, (200, 20)))               # 센서 offset과 잡음
y_sim, y_real = P[:, 0], P_real[:, 0]                             # 목표: 감쇠비 ζ

def rmse(model, X, y):
    return np.sqrt(np.mean((model.predict(X) - y) ** 2))

model = Ridge(alpha=1e-3).fit(X_sim[:300], y_sim[:300])          # 시뮬레이션만으로 학습
print(f"ζ RMSE: sim test {rmse(model, X_sim[300:], y_sim[300:]):.3f}, real {rmse(model, X_real[50:], y_real[50:]):.3f}")

X_dom = np.vstack([X_sim[:200], X_real])                          # (400, 20) sim과 real을 섞는다
d_dom = np.r_[np.zeros(200), np.ones(200)]                        # 0 = sim, 1 = real
auc = cross_val_score(LogisticRegression(max_iter=1000), X_dom, d_dom, cv=5, scoring="roc_auc").mean()
print(f"sim과 real을 구별하는 분류기의 AUC: {auc:.2f}")
```

```output
ζ RMSE: sim test 0.040, real 0.295
sim과 real을 구별하는 분류기의 AUC: 0.91
```

시뮬레이션 test에서 $\zeta$ RMSE 0.040이던 모델이 실측에서 0.295로 일곱 배 넘게 나빠졌다. $\zeta$의 범위가 0.05–0.5이므로 실측에서는 거의 쓸 수 없는 수준이다. 출처 분류기의 AUC 0.91은 두 데이터가 쉽게 구별될 만큼 다르다는 것을 label 없이도 보여 준다. 어떤 특징(궤적의 어느 시점)이 구별에 쓰였는지 보면 시뮬레이터가 놓친 현상을 찾는 단서가 된다.

::quiz{id="domain-classifier-auc"}

**간극을 줄이는 방법.** 세 가지 전략을 비교한다.

- **혼합 학습**: 시뮬레이션 데이터에 label이 있는 소량의 실측 데이터를 섞어 학습한다.
- **실측만**: 소량의 실측 데이터만으로 학습한다(비교 기준).
- **domain randomization**: 시뮬레이터가 모르는 요소(여기서는 비선형 강성 $\alpha$, 센서 offset)를 넓은 범위에서 무작위로 바꾸며 시뮬레이션해, 실제 계가 그 변동의 한 사례가 되게 한다. 로봇공학에서 시뮬레이션으로 학습한 시각 모델을 실제 세계로 옮기기 위해 제안된 방법이다(Tobin 외, 2017).

```python {exec="browser" session="synth"}
X_r20, y_r20 = X_real[:20], y_real[:20]                           # 라벨이 있는 실측 20개
mixed = Ridge(alpha=1e-3).fit(np.vstack([X_sim[:300], X_r20]), np.r_[y_sim[:300], y_r20])
real_only = Ridge(alpha=1e-3).fit(X_r20, y_r20)
alpha_rand = rng.uniform(0.0, 4.0, 300)                           # (300,) 모르는 물리를 무작위로
X_dr = (simulate(P[:300, 0], P[:300, 1], alpha=alpha_rand)
        + rng.uniform(-0.1, 0.1, (300, 1)) + rng.normal(0, 0.02, (300, 20)))
randomized = Ridge(alpha=1e-3).fit(X_dr, y_sim[:300])
for name, m in [("sim + 실측 20", mixed), ("실측 20만", real_only), ("domain randomization", randomized)]:
    print(f"{name:20s} real ζ RMSE {rmse(m, X_real[50:], y_real[50:]):.3f}")
```

```output
sim + 실측 20          real ζ RMSE 0.128
실측 20만               real ζ RMSE 0.104
domain randomization real ζ RMSE 0.065
```

이 예에서는 domain randomization이 가장 좋았다(0.065). 실제 계의 $\alpha = 1$과 offset 0.02가 무작위화한 범위($\alpha \in [0, 4]$, offset $\in [-0.1, 0.1]$) 안에 있었기 때문이다. 시뮬레이션 300개에 실측 20개를 섞은 혼합 학습(0.128)은 시뮬레이션만(0.295)보다 나았지만 실측 20개만 쓴 모델(0.104)보다 나빴다. 시뮬레이션 데이터가 압도적으로 많아 실측의 정보가 묻혔다. 혼합할 때는 실측 샘플에 가중치를 주거나, 시뮬레이션으로 사전학습한 뒤 실측으로 미세조정하는 방법을 함께 비교한다.

결과를 일반화하지 않도록 주의한다. domain randomization이 통한 것은 **무엇을 무작위화할지 알았기 때문**이다. 실제 간극의 원인이 무작위화한 요소 밖에 있으면 효과가 없다. 어떤 전략이 나은지는 문제마다 다르므로, 소량이라도 실측 test를 따로 두고 모든 전략을 그 위에서 비교한다. 이 실측 test는 전략 선택에 쓰는 순간 validation이 된다(15.4장). 최종 보고용 실측 test는 따로 남겨 둔다.

::quiz{id="sim-only-claim"}

:::note[분야 예시]
로봇 조작에서는 물체의 질감·조명·마찰 계수를 무작위화한 시뮬레이션으로 학습한 정책을 실제 로봇에 옮긴다. 전산유체역학(CFD) 데이터로 학습한 대리 모델(surrogate)은 격자 해상도와 난류 모델의 가정을 함께 배우므로, 풍동 실험 같은 실측과 비교해 간극을 잰다. 배터리 전기화학 모델(예: 단일 입자 모델)로 만든 데이터는 노화 메커니즘의 일부만 반영하므로, 실제 셀의 사이클 데이터로 검증하지 않은 수명 예측은 시뮬레이터의 가정 안에서만 의미가 있다.
:::

## 실습

시뮬레이터로 Latin hypercube 기반 데이터셋을 생성하고, 실제 데이터와의 분포 차이를 측정한다.

1. **설정 파일.** 시뮬레이터(자신의 연구 모델, 또는 이 장의 감쇠 진동자)의 파라미터 범위와 근거, 초기조건, 입력 프로파일, 설계(LHS), 표본 수, 시드, solver 설정을 하나의 설정 파일로 쓴다. 생성 스크립트가 설정 해시와 출력 manifest 해시를 기록하게 한다.
2. **설계 비교.** 같은 실행 수로 grid·random·LHS·Sobol을 만들어 축별 커버리지와 discrepancy를 비교한다. 아래 연습문제로 LHS를 직접 구현한다.
3. **간극 측정.** 실측 데이터(없으면 이 장처럼 시뮬레이터가 모르는 효과를 넣은 "실제 계")로 성능 간극과 출처 분류기 AUC를 잰다.
4. **간극 줄이기.** 혼합 학습, 실측만, domain randomization 중 둘 이상을 실측 validation에서 비교하고, 선택한 전략을 별도의 실측 test에서 한 번 평가한다.
5. **재생성 확인.** 다른 사람이 설정 파일과 코드 커밋만으로 같은 데이터셋(같은 manifest 해시)을 만드는지 확인한다.

::exercise{id="latin-hypercube"}

## 실패 사례

- **시뮬레이터의 기본 파라미터 하나로 생성한 데이터로 일반화를 주장하는 것.** 모델은 그 한 설정의 시뮬레이터를 배웠을 뿐이다. 파라미터 범위를 설계로 덮고, 실측으로 검증한다.
- **시뮬레이션 test만으로 성능을 보고하는 것.** 학습과 test가 같은 가정을 공유하므로 간극이 보이지 않는다. 실측 test를 따로 둔다.
- **설정을 기록하지 않은 생성.** 같은 데이터를 다시 만들 수 없고, 어떤 범위와 가정에서 나온 데이터인지 알 수 없다.
- **grid로 고차원 공간을 덮으려는 것.** 실행 수가 $k^d$로 폭발하고, 축마다 몇 개 값만 시험한다. LHS·Sobol을 쓴다.
- **시뮬레이션 데이터를 압도적으로 많이 섞고 실측의 효과를 기대하는 것.** 실측의 정보가 묻힌다. 가중치나 미세조정을 비교한다.

## 통과 기준

- 데이터셋 생성 설정(시뮬레이터 버전, 파라미터 범위와 근거, 설계, 시드, solver 설정)을 문서화하고, 다른 사람이 같은 manifest 해시의 데이터셋을 다시 만든다.
- 같은 실행 수의 설계들을 축별 커버리지와 discrepancy로 비교해 선택 근거를 쓴다.
- 성능 간극과 출처 분류기 AUC로 sim-to-real 간극을 측정하고, 간극을 줄이는 전략을 실측 데이터에서 비교한다.
- 연습문제 `latin-hypercube`의 테스트를 통과한다.

## 참고 문헌

- McKay, Beckman, Conover (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics*, 21(2), 239–245. — 컴퓨터 실험의 입력 선택에서 Latin hypercube sampling의 도입
- Tobin, Fong, Ray, Schneider, Zaremba, Abbeel (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. arXiv:1703.06907. — 시뮬레이터 요소를 무작위화해 실제 세계로 옮기는 방법
