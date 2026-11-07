---
id: chemistry-and-materials/interatomic-potentials
title: "기계학습 원자간 퍼텐셜"
title_en: "Machine-Learned Interatomic Potentials"
tier: advanced
status: draft
prerequisites: [chemistry-and-materials/molecular-representations, scientific-ml/geometric-models]
---

# 기계학습 원자간 퍼텐셜

분자동역학(molecular dynamics, MD)은 원자에 작용하는 힘을 반복해서 계산해 시간에 따른 운동을 적분하는 시뮬레이션이다. 힘을 양자화학 계산(DFT 등)으로 매 스텝 구하면 정확하지만 느려서, 큰 계나 긴 시간은 다루기 어렵다. **기계학습 원자간 퍼텐셜**(machine-learned interatomic potential, MLIP)은 양자화학 계산 데이터로 "원자 배치 → 에너지·힘" 함수를 학습해, DFT에 가까운 정확도로 수천 배 빠르게 힘을 내는 것을 목표로 한다.

이 장은 등변 신경망이나 기하 모델의 구조를 다시 가르치지 않는다. 그것은 7.10장(대칭), 18.6장(GNN), 24.8장(등변·기하 모델)에 있다. 여기서 가르치는 판단은 두 가지다. 하나는 **힘이 에너지의 gradient**($\mathbf{F} = -\nabla E$)라는 관계가 왜 학습에 중요한지, 다른 하나는 **정적인 테스트셋 오차와 동역학적 안정성이 왜 다른 것**인지다. 둘 다 Lennard-Jones 퍼텐셜이라는 작은 모형으로 직접 계산해 확인한다. 실제 MLIP는 복잡하지만, 여기서 보는 원리는 그대로 적용된다.

## 학습 목표

- 에너지·힘을 학습하는 퍼텐셜을 이해한다
- 분자동역학에서의 안정성과 외삽을 평가한다

## 문제 정의

**왜 필요한가.** MD의 한 스텝은 뉴턴의 운동 방정식 $m\ddot{\mathbf{r}}_i = \mathbf{F}_i$를 푼다. 힘만 있으면 적분할 수 있지만, 힘이 어떤 에너지의 gradient여야 물리가 지켜진다. 이 관계를 모델에 어떻게 넣느냐가 MLIP 설계의 핵심 결정이다.

**정의.** 계의 **퍼텐셜 에너지** $E(\mathbf{R})$는 모든 원자 좌표 $\mathbf{R} \in \mathbb{R}^{n \times 3}$의 함수다. 원자 $i$에 작용하는 힘은 에너지의 음의 gradient다.

$$
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}.
$$

7.10장에서 본 대로 에너지는 회전·평행이동·같은 원소의 순열에 **불변**(스칼라)이고, 불변 함수를 미분한 힘은 회전에 **등변**(벡터)이다. 그래서 MLIP의 표준 설계는 **불변인 에너지 하나를 신경망으로 출력하고, 힘은 그 에너지를 입력 좌표로 자동미분**해서 얻는 것이다(Behler, Parrinello, 2007). 이렇게 하면 힘이 정의상 어떤 에너지의 gradient가 되어, 보존력(conservative force)이 보장된다.

**Lennard-Jones 모형.** 가장 단순한 쌍 퍼텐셜인 Lennard-Jones(LJ)로 에너지와 힘을 직접 구현한다. 원자 쌍 $(i, j)$의 거리가 $r$일 때 에너지는 $4\varepsilon[(\sigma/r)^{12} - (\sigma/r)^6]$이고, 전체 에너지는 모든 쌍의 합이다. 힘은 이 에너지를 좌표로 미분한 해석적 식이다. 해석적 힘이 수치 미분(중심 차분)과 맞는지 확인하는 것은, 자동미분으로 힘을 구하는 MLIP에서 gradient 구현을 검증하는 바로 그 절차다. 단위는 LJ 환산 단위($\varepsilon = \sigma = m = 1$)를 쓴다.

```python {exec="browser" session="mlip"}
import numpy as np

def lj_energy(R):                                  # R (n, 3), 스칼라 에너지
    d = np.linalg.norm(R[:, None] - R[None], axis=-1)   # (n, n)
    iu = np.triu_indices(len(R), 1)                # 각 쌍을 한 번만
    inv6 = (1 / d[iu]) ** 6
    return np.sum(4 * (inv6 ** 2 - inv6))

def lj_forces(R):                                  # 해석적 힘 F = -dE/dR (n, 3)
    diff = R[:, None] - R[None]                    # (n, n, 3)
    d = np.linalg.norm(diff, axis=-1)              # (n, n)
    np.fill_diagonal(d, np.inf)                    # 자기 자신 제외
    inv6 = (1 / d) ** 6
    coeff = 24 * (2 * inv6 ** 2 - inv6) / d ** 2   # (n, n)
    return np.einsum("ij,ijk->ik", coeff, diff)    # 쌍마다 방향 벡터에 곱해 합

re = 2 ** (1 / 6)                                   # LJ 평형 거리
R = np.array([[0, 0, 0], [re, 0, 0], [-re, 0, 0], [0, re, 0],
              [0, -re, 0], [0, 0, re], [0, 0, -re]], float)   # 중심 + 팔면체 7원자
R = R + np.random.default_rng(0).normal(size=R.shape) * 0.05  # 평형에서 살짝 흔들기
F = lj_forces(R)
h = 1e-5                                            # 중심 차분으로 수치 힘
F_num = np.zeros_like(R)
for i in range(R.shape[0]):
    for k in range(3):
        Rp, Rm = R.copy(), R.copy()
        Rp[i, k] += h; Rm[i, k] -= h
        F_num[i, k] = -(lj_energy(Rp) - lj_energy(Rm)) / (2 * h)
print("에너지 E =", round(float(lj_energy(R)), 4))
print("해석적 힘과 수치 힘의 최대 차이 < 1e-6:", np.abs(F - F_num).max() < 1e-6)
print("전체 힘의 합(뉴턴 제3법칙) ~ 0:", np.abs(F.sum(0)).max() < 1e-10)
```

```output
에너지 E = -8.9017
해석적 힘과 수치 힘의 최대 차이 < 1e-6: True
전체 힘의 합(뉴턴 제3법칙) ~ 0: True
```

해석적 힘이 수치 미분과 맞고, 전체 힘의 합이 0이다(쌍 힘이 작용·반작용으로 상쇄되어 계 전체에 알짜 힘이 없다). 이 두 점검은 MLIP를 구현할 때 반드시 통과시켜야 하는 최소 조건이다.

## 학습 데이터

**왜 필요한가.** MLIP의 정확도는 학습 데이터가 덮는 원자 배치의 범위가 정한다. MD는 학습 데이터에 없던 배치를 끊임없이 만들어 내므로, 데이터가 덮지 못한 배치에서 힘이 틀리면 궤적이 비물리적으로 흐른다.

**계산 수준의 일관성.** 레이블은 양자화학 계산에서 온다. 계산 수준(함수, 기저 집합, 설정)이 섞이면 모델은 물리가 아니라 계산 설정의 차이를 배운다. 한 데이터셋은 한 수준으로 통일한다(28.5장).

**힘이 에너지보다 정보가 많다.** 한 번의 양자화학 계산은 에너지 하나와 모든 원자의 힘을 함께 준다. 에너지는 스칼라 1개지만, 힘은 원자마다 3개 성분이다. 그래서 힘까지 학습에 쓰면 같은 계산 비용으로 훨씬 많은 제약을 얻는다.

```python {exec="browser" session="mlip"}
n = 20
print(f"원자 {n}개 배치 하나에서 한 번의 양자화학 계산이 주는 제약:")
print(f"  에너지만 학습: 스칼라 {1}개")
print(f"  힘도 학습    : 에너지 1개 + 힘 {n} × 3 = {3 * n}개 성분")
print(f"  → 같은 계산 비용으로 힘은 배치마다 {3 * n + 1}배의 제약을 준다")
```

```output
원자 20개 배치 하나에서 한 번의 양자화학 계산이 주는 제약:
  에너지만 학습: 스칼라 1개
  힘도 학습    : 에너지 1개 + 힘 20 × 3 = 60개 성분
  → 같은 계산 비용으로 힘은 배치마다 61배의 제약을 준다
```

원자 20개면 힘은 60개 성분, 에너지 1개와 합쳐 배치마다 61개의 제약이다. 힘은 에너지 지형의 **기울기**를 직접 알려 주므로, 에너지만으로 지형의 모양을 추측하는 것보다 훨씬 효율적이다. 그래서 MLIP 학습의 손실은 대개 에너지 오차와 힘 오차의 가중합이고, 힘 항에 큰 가중치를 둔다. 적은 양자화학 계산으로 정확한 퍼텐셜을 만드는 방법들이 이 원리에 기댄다(Chmiela 외, 2017; Batzner 외, 2022).

::quiz{id="forces-more-informative"}

**능동 학습.** 학습 데이터를 한 번에 다 만들지 않고, MD를 돌리다 모델이 자신 없어 하는 배치를 만나면 그것만 양자화학으로 계산해 데이터에 추가하는 방식이 쓰인다. 여러 모델의 힘 예측이 크게 갈리는 배치(epistemic 불확실성이 큰 배치)를 고르는 query-by-committee가 대표적이다(Smith 외, 2018; 21.1장의 능동 학습과 같은 원리). 이렇게 하면 화학 공간을 무작위로 덮는 것보다 훨씬 적은 계산으로 안정적인 퍼텐셜을 만든다.

## 모델

**국소성 가정.** 대부분의 MLIP는 전체 에너지를 원자별 에너지의 합 $E = \sum_i E_i$로 쓰고, 각 원자 에너지 $E_i$는 반지름 cutoff 안의 이웃 원자들로만 정한다. 이 **국소성**(locality) 가정 덕분에 계의 크기에 선형으로 비례하는 계산으로 에너지를 구하고, 학습에 없던 더 큰 계로도 일반화한다. 대신 cutoff보다 먼 상호작용(장거리 정전기, 분산력)은 따로 더해야 한다.

**두 계열.** 7.10장과 24.8장의 언어로 두 설계를 구분한다.

- **불변 기술자 기반**: 각 원자의 국소 환경을 회전·순열 불변인 기술자(대칭 함수, 거리·각도 분포)로 바꾼 뒤 신경망에 넣는다(Behler, Parrinello, 2007). 입력이 이미 불변이라 구조가 단순하다.
- **등변 message passing**: 거리 같은 불변량과 상대 벡터 같은 등변량을 함께 전달하는 그래프 신경망이다(18.6장의 EGNN, 24.8장). 방향 정보를 층 사이에 유지해, 같은 데이터로 더 높은 정확도와 데이터 효율을 보인다(Batzner 외, 2022).

어느 계열이든 에너지를 출력하고 힘은 자동미분으로 얻는 것이 표준이다. 힘을 신경망의 별도 출력으로 직접 예측하면 더 빠를 수 있지만, 그 힘이 어떤 에너지의 gradient라는 보장이 사라진다. 다음 절에서 이 선택의 대가를 본다.

## 시뮬레이션에서의 평가

**velocity Verlet와 에너지 보존.** MD 적분기는 보통 velocity Verlet을 쓴다. 한 스텝은 힘으로 속도를 반 스텝 갱신하고, 위치를 갱신하고, 새 힘으로 속도를 마저 갱신한다. 이 적분기는 외력이 없는 계에서 총에너지(운동 + 퍼텐셜)를 시간 간격 $\Delta t$의 제곱에 비례하는 오차 안에서 보존한다. 참 힘(LJ)으로 적분하며 $\Delta t$를 줄일 때 에너지 표류가 $\Delta t^2$로 주는지 확인한다.

```python {exec="browser" session="mlip"}
def velocity_verlet(R, V, dt, steps):
    F = lj_forces(R)
    traj_E = []
    for _ in range(steps):
        V = V + 0.5 * dt * F                       # 반 스텝 속도
        R = R + dt * V                             # 위치 갱신
        F = lj_forces(R)                           # 새 힘
        V = V + 0.5 * dt * F                       # 나머지 반 스텝
        traj_E.append(0.5 * np.sum(V ** 2) + lj_energy(R))
    return np.array(traj_E)

R0 = np.array([[0.0, 0, 0], [1.2, 0, 0]])          # 두 원자, 평형 1.122보다 약간 멂
V0 = np.array([[0.1, 0, 0], [-0.1, 0, 0]])
E0 = 0.5 * np.sum(V0 ** 2) + lj_energy(R0)
prev = None
for dt in [0.05, 0.01, 0.002]:
    E = velocity_verlet(R0.copy(), V0.copy(), dt, int(2.0 / dt))
    drift = np.abs(E - E0).max()
    ratio = "" if prev is None else f"  (dt 1/5 → 표류 1/{prev / drift:.0f})"
    print(f"dt = {dt:.3f}  총에너지 표류 {drift:.2e}{ratio}")
    prev = drift
```

```output
dt = 0.050  총에너지 표류 7.83e-03
dt = 0.010  총에너지 표류 3.13e-04  (dt 1/5 → 표류 1/25)
dt = 0.002  총에너지 표류 1.25e-05  (dt 1/5 → 표류 1/25)
```

$\Delta t$를 1/5로 줄일 때마다 에너지 표류가 약 1/25로 준다. 2차 수렴이다. 보존력에서는 적분 오차가 이렇게 통제되므로, 에너지가 체계적으로 새면 적분기가 아니라 **힘** 쪽을 의심한다.

**비보존력의 대가.** 힘을 에너지의 gradient로 얻지 않고 직접 예측하는 모델은, 그 힘장이 어떤 에너지의 gradient가 아닐 수 있다. 그런 힘을 **비보존력**(non-conservative force)이라 한다. 비보존력은 닫힌 경로를 돌 때 알짜 일을 해서, MD에서 에너지를 체계적으로 넣거나 뺀다. 참 LJ 힘에 작은 회전형(비보존) 성분을 더해, 정적 배치에서의 힘 오차는 작은데도 MD가 어떻게 되는지 본다.

```python {exec="browser" session="mlip"}
cluster = np.array([[0, 0, 0], [re, 0, 0], [-re, 0, 0], [0, re, 0],
                    [0, -re, 0], [0, 0, re], [0, 0, -re]], float)

def run(force_fn, steps=4000, dt=0.004):
    R = cluster + np.random.default_rng(1).normal(size=cluster.shape) * 0.03
    V = np.zeros_like(R)
    F = force_fn(R)
    Es, size = [], []
    for _ in range(steps):
        V = V + 0.5 * dt * F; R = R + dt * V; F = force_fn(R); V = V + 0.5 * dt * F
        Es.append(0.5 * np.sum(V ** 2) + lj_energy(R))
        size.append(np.linalg.norm(R - R.mean(0), axis=1).max())
    return np.array(Es), np.array(size)

omega = np.array([0, 0, 1.0])
def rms_force_error(gain):                          # 참 힘 대비 추가된 힘의 RMS (평형 근처 배치에서)
    extra = gain * np.cross(omega, cluster - cluster.mean(0))
    return np.sqrt(np.mean(extra ** 2)) / np.sqrt(np.mean(lj_forces(cluster + 0.03) ** 2))

for gain in [0.0, 0.05, 0.15]:
    fn = (lambda R: lj_forces(R)) if gain == 0 else (lambda R, g=gain: lj_forces(R) + g * np.cross(omega, R - R.mean(0)))
    Es, size = run(fn)
    label = "참 힘(보존력)        " if gain == 0 else f"비보존력 (힘 RMS 오차 {rms_force_error(gain):.0%})"
    gained = Es[-1] - Es[0] > 0.1                   # 에너지가 유의미하게 늘었는가
    blew_up = size.max() > 2.0                      # 계가 평형 크기의 두 배 넘게 퍼졌는가
    print(f"{label}  에너지 증가 {gained!s:5s}  계가 터짐 {blew_up!s:5s}")
```

```output
참 힘(보존력)          에너지 증가 False  계가 터짐 False
비보존력 (힘 RMS 오차 2%)  에너지 증가 True   계가 터짐 False
비보존력 (힘 RMS 오차 6%)  에너지 증가 True   계가 터짐 True 
```

힘 RMS 오차가 2%일 때 이미 에너지가 체계적으로 늘고, 6%면 계가 평형 크기의 두 배 넘게 퍼져 터진다. **정적 배치에서의 힘 오차는 몇 %로 작은데도** MD는 불안정하다. 작은 오차라도 비보존 성분이 있으면 긴 궤적에서 누적되기 때문이다. 이것이 힘을 에너지의 gradient로 얻는 설계를 쓰는 실질적 이유이고, 힘을 직접 예측하는 더 빠른 모델이 에너지 보존 점검을 반드시 통과해야 하는 이유다.

::quiz{id="non-conservative"}

**정적 오차와 동역학 안정성은 다르다.** 위 실험의 교훈을 일반화하면, 테스트셋의 힘 MAE가 작다고 MD가 안정하다는 보장은 없다. 두 평가는 서로 다른 것을 잰다.

- **정적 오차**: 학습 분포에서 뽑은 배치에서의 에너지·힘 MAE. 보간 성능이다.
- **동역학 안정성**: MD를 길게 돌렸을 때 궤적이 비물리로 가지 않는가. 모델은 자기 예측이 만든 새 배치를 다시 입력으로 받으므로(자기회귀), 작은 오차가 누적되고 학습 분포 밖으로 나간다.

힘 오차가 흔히 보고되는 지표지만 시뮬레이션의 쓸모와는 잘 맞지 않고, 안정성이 별도의 핵심 지표라는 점이 벤치마크 연구에서 보고되었다(Fu 외, 2023). 그래서 MLIP는 정적 오차와 함께 **MD 안정성**(터지지 않고 돌아가는 시간), **관측량 재현**(동경 분포 함수, 확산 계수, 상전이 온도 같은 실제 관심 물리량)으로 평가한다. 정적 오차가 작은 두 모델이 같은 MD에서 전혀 다른 거동을 보일 수 있다.

:::note[분야 예시]
물의 MLIP를 만들 때 목표는 힘 MAE를 줄이는 것 자체가 아니라, MD에서 물의 밀도, 확산 계수, 동경 분포 함수, 끓는점 같은 관측량을 재현하는 것이다. 힘 오차가 비슷한 두 모델이 확산 계수를 두 배 차이로 낼 수 있어서, 정적 지표만으로는 어느 쪽이 쓸 만한지 고를 수 없다. 재료에서도 MLIP로 녹는점이나 상전이를 예측할 때는, 그 전이 근처의 배치가 학습 데이터에 충분히 있는지와 긴 MD가 안정한지를 함께 확인한다.
:::

## 실습

작은 LJ 계의 퍼텐셜을 다루며 힘의 검증, velocity Verlet 적분, 에너지 보존을 함께 평가한다. 본문 코드가 그 전체 흐름을 보여 준다. 실습에서는 세 가지를 바꿔 본다.

- **적분 시간 간격**을 키우며 에너지 표류가 2차로 늘다가, 어느 $\Delta t$를 넘으면 계가 터지는 임계를 찾는다.
- **초기 속도(온도)**를 높이며 계가 결합을 유지하는 한계를 본다. 학습 데이터가 낮은 온도만 덮었다면, 높은 온도의 배치는 외삽이다.
- **비보존 힘 성분의 세기**를 바꾸며 정적 힘 오차와 MD 안정성의 관계를 그린다. 둘이 단순 비례하지 않음을 확인한다.

두 연습문제를 푼다. 하나는 LJ 힘의 해석적 식을, 다른 하나는 velocity Verlet 한 스텝을 구현한다. 테스트가 수치 미분과 에너지 보존으로 채점한다.

::exercise{id="lj-forces"}

::exercise{id="velocity-verlet-step"}

## 실패 사례

- **테스트셋 힘 오차가 작으면 MD도 안정하다고 가정하는 것.** 정적 오차와 동역학 안정성은 다른 것을 잰다. 모델은 자기 예측이 만든 배치를 다시 입력으로 받으므로 작은 오차가 누적된다. 안정성과 관측량을 따로 평가한다.
- **힘을 직접 예측하면서 에너지 보존을 확인하지 않는 것.** 비보존 힘은 정적 오차가 작아도 MD에서 에너지를 체계적으로 넣거나 빼 계를 터뜨린다. 외력 없는 계에서 총에너지가 보존되는지 점검한다.
- **학습 데이터가 덮지 못한 배치로 외삽하는 것.** 낮은 온도만 학습한 퍼텐셜을 고온 MD에 쓰거나, 평형 근처만 학습하고 결합이 끊기는 반응 경로를 시뮬레이션하면 힘이 틀린다. MD가 만드는 배치가 학습 분포 안인지 불확실성으로 감시한다.
- **계산 수준이 다른 레이블을 섞는 것.** 모델이 물리가 아니라 계산 설정의 차이를 배운다.

## 통과 기준

- 퍼텐셜을 정적 오차(에너지·힘 MAE)와 동역학 안정성(MD가 터지지 않고 돌아가는 시간, 관측량 재현)으로 **함께** 보고한다.
- 외력 없는 계에서 총에너지가 보존되는지, $\Delta t$를 줄일 때 표류가 줄어드는지 확인한 결과를 보고한다.
- 연습문제 `lj-forces`, `velocity-verlet-step`의 테스트를 통과한다.

## 참고 문헌

- Behler, Parrinello (2007). Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces. *Physical Review Letters*, 98, 146401. — 국소 환경 기반 신경망 퍼텐셜, 에너지에서 힘을 얻는 설계
- Chmiela, Tkatchenko, Sauceda, Poltavsky, Schütt, Müller (2017). Machine learning of accurate energy-conserving molecular force fields. *Science Advances*, 3, e1603015. — 에너지 보존을 보장하는 gradient-domain 학습
- Smith, Nebgen, Lubbers, Isayev, Roitberg (2018). Less is more: Sampling chemical space with active learning. *Journal of Chemical Physics*, 148, 241733. — query-by-committee 능동 학습으로 데이터 수집
- Batzner, Musaelian, Sun, Geiger, Mailoa, Kornbluth, Molinari, Smidt, Kozinsky (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13, 2453. — 등변 message passing 퍼텐셜(NequIP)과 데이터 효율
- Fu, Wu, Wang, Xie, Keten, Gómez-Bombarelli, Jaakkola (2023). Forces are not Enough: Benchmark and Critical Evaluation for Machine Learning Force Fields with Molecular Simulations. *Transactions on Machine Learning Research*. — 힘 오차와 시뮬레이션 안정성의 불일치, 안정성 지표의 중요성
