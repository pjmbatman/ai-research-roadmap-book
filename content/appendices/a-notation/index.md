---
id: appendices/notation
title: "표기법"
title_en: "Notation"
tier: reference
status: draft
prerequisites: []
---

# 표기법

이 부록은 책 전체에서 쓰는 수학 기호와 코드 표기를 한곳에 모은다. 장마다 이 표기를 따르며, 어떤 장이 다른 표기를 써야 할 때는 그 장 첫머리의 **지역 표기 상자**에 차이만 적는다.

## 학습 목표

- 책 전체의 수학·코드 표기 규칙을 한곳에서 확인한다
- 분야마다 다른 기호가 충돌할 때 이 책이 어떤 쪽을 따르는지 안다

## 수와 집합

| 표기 | 뜻 |
|---|---|
| $\mathbb{R}, \mathbb{R}^d, \mathbb{R}^{m \times n}$ | 실수, $d$차원 실수 벡터, $m \times n$ 실수 행렬 |
| $\mathbb{N}, \mathbb{Z}$ | 자연수, 정수 |
| $\{1, \dots, N\}$ 또는 $[N]$ | 1부터 $N$까지의 정수 |
| $\mathcal{A}, \mathcal{X}$ | 집합·공간 (필기체 대문자) |
| $\lvert \mathcal{A} \rvert$ | 집합의 원소 수 |
| $a \in \mathcal{A}$, $\mathcal{A} \subseteq \mathcal{B}$ | 원소, 부분집합 |
| $:=$ | 정의 |
| $\approx$ | 근사적으로 같다 |
| $\propto$ | 비례한다 (정규화 상수를 생략) |

## 스칼라·벡터·행렬·텐서

| 표기 | 뜻 |
|---|---|
| $x, \alpha, \lambda$ | 스칼라 (이탤릭 소문자, 그리스 문자) |
| $\mathbf{x} \in \mathbb{R}^d$ | 벡터. **열벡터**가 기본이다 |
| $x_i$ | 벡터 $\mathbf{x}$의 $i$번째 성분 |
| $\mathbf{A} \in \mathbb{R}^{m \times n}$ | 행렬 (굵은 대문자) |
| $A_{ij}$ | 행렬 $\mathbf{A}$의 $(i, j)$ 성분 |
| $\mathbf{A}^\top$ | 전치 |
| $\mathbf{A}^{-1}$, $\mathbf{A}^{+}$ | 역행렬, pseudoinverse |
| $\mathbf{I}$, $\mathbf{I}_n$ | 단위행렬 |
| $\mathbf{0}$, $\mathbf{1}$ | 영벡터, 모든 성분이 1인 벡터 |
| $\operatorname{diag}(\mathbf{v})$ | 대각 성분이 $\mathbf{v}$인 대각행렬 |
| $\operatorname{tr}(\mathbf{A})$, $\det(\mathbf{A})$, $\operatorname{rank}(\mathbf{A})$ | 대각합, 행렬식, 계수 |
| $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y}$ | 내적 |
| $\lVert \mathbf{x} \rVert_2$, $\lVert \mathbf{x} \rVert_1$, $\lVert \mathbf{A} \rVert_F$ | L2 노름, L1 노름, Frobenius 노름. 아래첨자가 없으면 L2 |
| $\mathbf{x} \odot \mathbf{y}$ | 성분별 곱 (Hadamard) |
| $\mathsf{X}$ | 3차원 이상의 텐서 (산세리프 대문자). 코드에서는 shape 주석으로 대신한다 |

## 함수와 미분

| 표기 | 뜻 |
|---|---|
| $f: \mathcal{X} \to \mathcal{Y}$ | 정의역과 공역 |
| $f \circ g$ | 합성 $f(g(\cdot))$ |
| $\frac{df}{dx}$, $f'(x)$ | 일변수 미분 |
| $\frac{\partial f}{\partial x_i}$ | 편미분 |
| $\nabla_{\mathbf{x}} f \in \mathbb{R}^d$ | gradient. $f: \mathbb{R}^d \to \mathbb{R}$에 대해 **열벡터** |
| $\mathbf{J}_f \in \mathbb{R}^{m \times n}$ | Jacobian. $f: \mathbb{R}^n \to \mathbb{R}^m$에 대해 (출력 차원 × 입력 차원) |
| $\nabla^2 f$, $\mathbf{H}$ | Hessian |
| $\dot{x}$, $\ddot{x}$ | 시간에 대한 1계·2계 도함수 (물리 예제) |
| $\partial_t u$, $\partial_{xx} u$ | 편미분의 축약 (PDE 예제) |

Jacobian과 gradient의 관계는 $\nabla_{\mathbf{x}} f = \mathbf{J}_f^\top$이다($m = 1$일 때). 다른 교재에서 gradient를 행벡터로 쓰는 경우(numerator layout)가 있으므로, 공식을 옮겨 올 때 전치에 주의한다.

## 확률과 통계

| 표기 | 뜻 |
|---|---|
| $X, Y$ | 확률변수 (대문자) |
| $x, y$ | 확률변수가 취한 값 (소문자) |
| $p(x)$ | 확률질량함수 또는 확률밀도함수. 문맥으로 구분한다 |
| $p(x, y)$, $p(y \mid x)$ | 결합분포, 조건부분포 |
| $p_\theta(y \mid x)$ | 파라미터 $\theta$를 가진 모델의 조건부분포 |
| $P(A)$, $\Pr[A]$ | 사건 $A$의 확률 |
| $X \sim p$ | $X$가 분포 $p$를 따른다 |
| $\mathbb{E}[X]$, $\mathbb{E}_{x \sim p}[f(x)]$ | 기댓값. 어떤 분포에 대한 기댓값인지 아래첨자로 밝힌다 |
| $\operatorname{Var}[X]$, $\operatorname{Cov}[X, Y]$ | 분산, 공분산 |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | 평균 $\boldsymbol{\mu}$, 공분산 $\boldsymbol{\Sigma}$인 정규분포 |
| $\operatorname{Ber}(\pi)$, $\operatorname{Cat}(\boldsymbol{\pi})$, $\operatorname{Unif}(a, b)$ | Bernoulli, categorical, 균등분포 |
| $\hat{\theta}$ | $\theta$의 추정값 |
| $\theta^\star$ | 최적값 |
| $\mathbb{1}[\cdot]$ | 지시함수 (조건이 참이면 1) |
| $\mathrm{KL}(p \,\Vert\, q)$ | KL divergence |
| $H(p)$, $H(p, q)$ | 엔트로피, 교차엔트로피 |

## 학습 문제

| 표기 | 뜻 |
|---|---|
| $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ | 데이터셋. $N$은 샘플 수 |
| $\mathbf{x}$, $y$ | 입력, 출력(정답). 출력이 벡터면 $\mathbf{y}$ |
| $\hat{y} = f_\theta(\mathbf{x})$ | 모델의 점 예측 |
| $\theta$ | 학습되는 파라미터 전체 |
| $\mathbf{W}, \mathbf{b}$ | 한 층의 가중치 행렬과 bias |
| $\mathbf{z}$ | 잠재변수 |
| $\ell(\hat{y}, y)$ | 샘플 하나의 손실 |
| $L(\theta)$ | 전체 손실(목적함수) |
| $\eta$ | 학습률 |
| $\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{val}}, \mathcal{D}_{\text{test}}$ | 학습·검증·시험 데이터 |

## 시간과 동역학

| 표기 | 뜻 |
|---|---|
| $t$ | 연속 시간 |
| $k$ | 이산 시간 단계 (예: $\mathbf{x}_k = \mathbf{x}(k \Delta t)$) |
| $\Delta t$, $h$ | 시간 간격, 수치해법의 step 크기 |
| $\mathbf{s}(t)$ | 동역학 시스템의 상태 (ML의 입력 $\mathbf{x}$와 구분할 때) |
| $\mathbf{u}(t)$ | 제어 입력 |
| $u(x, t)$ | PDE의 해(장, field). 공간 좌표 $x$와 시간 $t$의 함수 |

**분야 간 충돌.** 제어·동역학 교재는 상태를 $\mathbf{x}$로 쓰고, ML 교재는 입력을 $\mathbf{x}$로 쓴다. 이 책은 ML 표기를 기본으로 하고, 동역학이 중심인 장은 상태를 $\mathbf{s}$로 쓰거나 지역 표기 상자에서 $\mathbf{x}$를 상태로 다시 정의한다.

## 공통 예제의 기호

| 예제 | 식 | 기호 |
|---|---|---|
| 실 1. 감쇠 진동자 | $\ddot{x} + 2\zeta\omega_0\dot{x} + \omega_0^2 x = 0$ | $x(t)$ 위치, $\zeta$ 감쇠비, $\omega_0$ 고유진동수 |
| 실 2. 열 방정식 | $\partial_t u = \kappa\,\partial_{xx} u$ | $u(x, t)$ 온도, $\kappa$ 열확산계수 |
| 실 3. 배터리 사이클 데이터 | — | $Q_k$ 사이클 $k$의 방전 용량, $\mathrm{SOH}_k = Q_k / Q_0$ |

## 코드 표기

텐서 연산 옆에는 shape 주석을 단다.

```python
h = encoder(x)        # (B, T, D)
attn = q @ k.mT       # (B, H, T, T)
```

| 축 약어 | 뜻 |
|---|---|
| `B` | batch |
| `T` | 시간 또는 시퀀스 길이 |
| `C` | 채널 |
| `D` | 특징 차원 |
| `H` | attention head 수 |
| `N` | 샘플 수 |

물리량을 담는 변수에는 단위 접미사를 붙인다.

| 접미사 | 단위 | 예 |
|---|---|---|
| `_s` | 초 | `time_s` |
| `_v` | 볼트 | `voltage_v` |
| `_a` | 암페어 | `current_a` |
| `_ah` | 암페어시 | `capacity_ah` |
| `_c` | 섭씨 | `temp_c` |
| `_ohm` | 옴 | `r0_ohm` |
| `_m`, `_kg` | 미터, 킬로그램 | `length_m`, `mass_kg` |
