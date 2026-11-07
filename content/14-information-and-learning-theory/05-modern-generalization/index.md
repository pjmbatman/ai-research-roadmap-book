---
id: information-and-learning-theory/modern-generalization
title: "현대 딥러닝의 일반화와 스케일링"
title_en: "Generalization and Scaling in Modern Deep Learning"
tier: advanced
status: draft
prerequisites: [information-and-learning-theory/generalization-basics]
---

# 현대 딥러닝의 일반화와 스케일링

14.4장의 고전적 그림은 분명했다. 모델이 복잡할수록 variance가 커지고, 학습 데이터를 완벽히 맞추는 모델은 잡음을 외운 것이므로 일반화가 나쁘다. 그런데 현대 신경망은 파라미터가 학습 데이터보다 훨씬 많고, 학습 오차를 0으로 만들고도 잘 일반화한다. 무작위 레이블까지 외울 수 있는 같은 신경망이 진짜 레이블에서는 좋은 시험 성능을 낸다(14.4장). 고전적 그림의 무엇이 틀렸는가.

이 장은 그 답의 조각들을 다룬다. 모델을 데이터보다 크게 키우면 시험 오차가 다시 내려가는 **double descent**, 같은 데이터를 완벽히 맞추는 수많은 해 중 학습 알고리즘이 특정한 해를 고르는 **암묵적 정규화**, 무한히 넓은 신경망을 분석하는 NTK와 feature learning의 관점, 그리고 모델·데이터·계산량을 키울 때 성능이 어떻게 변하는지 기술하는 **스케일링 법칙**이다. 마지막으로, 과학 문제에서 특히 중요한 보간과 외삽의 구분을 다시 본다. 이 분야의 이론은 아직 완성되지 않았다. 확립된 결과와 진행 중인 논쟁을 구분해서 읽어야 한다.

## 학습 목표

- 과매개변수화 모델의 일반화 현상을 안다
- 스케일링 법칙을 해석한다
- 보간과 외삽을 구분한다

## 과매개변수화

**왜 필요한가.** 실험 설계에서 "모델을 얼마나 크게 할 것인가"는 매번 내리는 결정이다. 고전적 U자 곡선을 믿으면 데이터 수보다 파라미터가 많은 모델은 피해야 한다. 그러나 현대 딥러닝의 표준 관행은 그 반대이고, 잘 동작한다. 두 그림을 하나로 이해해야 모델 크기를 근거를 갖고 고를 수 있다.

**정의.** 파라미터 수 $p$가 학습 표본 수 $N$보다 많아 학습 데이터를 정확히 맞추는 해가 존재하는 영역을 **과매개변수화**(overparameterized) 영역이라 한다. 학습 오차가 0이 되는 경계 $p \approx N$을 **보간 임계점**(interpolation threshold)이라 한다. 보간 임계점 너머에서는 학습 데이터를 정확히 맞추는 해가 무수히 많다. 그중 어떤 해를 고르는지가 일반화를 정한다.

**최소 노름 회귀.** 이 현상을 가장 단순하게 재현하는 모델은 선형 회귀다. 입력 특징이 $D = 200$개 있고, 모델은 앞의 $p$개만 쓴다. 표본은 $N = 40$개다. $p < N$이면 보통의 최소제곱 해를, $p > N$이면 학습 데이터를 정확히 맞추는 해 중 노름이 가장 작은 해 $\hat{\mathbf{w}} = \mathbf{X}^{+}\mathbf{y}$(유사역행렬, 7.6장)를 쓴다. 두 경우 모두 `np.linalg.pinv`가 계산한다. 참 계수는 앞쪽 특징일수록 크지만($\beta_j \propto j^{-1/2}$) 200개 모두에 퍼져 있다. 입력이 등방 가우시안이면 시험 MSE를 $\lVert\hat{\mathbf{w}} - \boldsymbol\beta\rVert^2 + \sigma^2$로 정확히 계산할 수 있다. 20번 반복한 중앙값을 본다. 비교를 위해 ridge($\lambda = 1$) 해도 함께 계산한다.

```python {exec="browser" session="modern"}
import numpy as np

n, D, noise = 40, 200, 0.1                         # 학습 표본 40개, 후보 특징 200개
beta = np.arange(1, D + 1) ** -0.5                 # 참 계수: 앞쪽 특징일수록 중요, (D,)
beta /= np.linalg.norm(beta)

def risk(w, p):                                    # 등방 가우시안 입력에서 시험 MSE의 정확한 값
    return np.sum((w - beta[:p]) ** 2) + np.sum(beta[p:] ** 2) + noise**2

print("  p    최소 노름: 위험   ‖w‖     ridge(λ=1): 위험")
for p in (5, 10, 20, 30, 36, 40, 44, 50, 80, 120, 200):
    out = []
    for seed in range(20):
        r = np.random.default_rng(seed)
        X = r.normal(size=(n, D))                  # (N, D)
        y = X @ beta + noise * r.normal(size=n)
        Xp = X[:, :p]                              # 앞의 p개 특징만 쓰는 모델, (N, p)
        w = np.linalg.pinv(Xp) @ y                 # 최소 노름 최소제곱 해, (p,)
        w_r = Xp.T @ np.linalg.solve(Xp @ Xp.T + 1.0 * np.eye(n), y)   # ridge 해, (p,)
        out.append((risk(w, p), np.linalg.norm(w), risk(w_r, p)))
    m = np.median(out, axis=0)
    print(f"{p:4d}       {m[0]:7.2f}      {m[1]:5.2f}       {m[2]:5.2f}")
```

```output
  p    최소 노름: 위험   ‖w‖     ridge(λ=1): 위험
   5          0.71       0.70        0.70
  10          0.67       0.82        0.67
  20          0.84       0.99        0.77
  30          1.47       1.41        0.98
  36          3.39       1.97        1.21
  40         24.91       4.99        1.20
  44          2.43       1.73        1.16
  50          1.39       1.24        0.97
  80          0.77       0.76        0.76
 120          0.76       0.60        0.75
 200          0.82       0.45        0.82
```

최소 노름 해의 위험은 세 구간을 지난다.

- **$p < N$**: 고전적 U자. $p = 10$에서 최소(0.67)이고, 특징을 더 넣으면 variance가 커진다.
- **$p = N = 40$**: 위험이 폭발한다(24.91). 방정식과 미지수의 수가 같아 데이터를 정확히 맞추는 해가 하나뿐이고, 그 해를 얻으려면 $\mathbf{X}$의 가장 작은 특잇값으로 잡음을 나누어야 한다. 정사각 무작위 행렬의 가장 작은 특잇값은 0에 가까우므로 잡음이 크게 증폭된다. 해의 노름(4.99)이 함께 치솟는 것이 그 흔적이다.
- **$p > N$**: 위험이 **다시 내려간다**(0.76까지). 보간하는 해가 많아지고, 그중 노름이 가장 작은 해는 특징이 늘수록 노름이 줄어든다(0.45까지).

이것이 **double descent**다(Belkin 외, 2019). 첫 번째 하강은 고전적 영역, 봉우리는 보간 임계점, 두 번째 하강은 과매개변수화 영역이다. 같은 현상이 random feature 모델, 결정 트리 앙상블, 신경망에서 관찰되었고(Belkin 외, 2019; Nakkiran 외, 2020), 선형 모델에서는 이론적으로 정확히 분석되어 있다(Hastie 외, 2022).

::quiz{id="double-descent-peak"}

결과를 읽을 때 주의할 점이 세 가지다.

- **두 번째 하강이 언제나 첫 번째 최솟값보다 낮은 것은 아니다.** 이 예에서 과매개변수화 영역의 최소(0.76)는 $p = 10$의 0.67보다 높다. 어느 쪽이 나은지는 신호가 특징에 어떻게 퍼져 있는지와 잡음 크기에 달려 있다.
- **봉우리는 정규화로 사라진다.** ridge($\lambda = 1$) 열에는 봉우리가 없다(최대 1.21). 봉우리는 "보간하는 해를 정규화 없이 고르는" 특정한 추정량의 성질이지, 모델 크기 자체의 저주가 아니다.
- **"복잡도"는 파라미터 수가 아니다.** 과매개변수화 영역에서 파라미터 수는 늘지만 해의 노름은 준다. 14.4장에서 본 노름 기반 용량으로 재면, 큰 모델이 오히려 더 "단순한" 해를 고르고 있다.

::widget{name="double-descent"}

::quiz{id="ridge-removes-peak"}

**신경망에서는.** Nakkiran 외(2020)는 신경망에서도 모델 폭을 키우며 시험 오차의 봉우리와 두 번째 하강을 관찰했고, 학습 epoch 수에 대해서도 같은 모양(epoch-wise double descent)을 보고했다. 레이블 잡음이 있을 때 봉우리가 뚜렷해진다. 또 봉우리 근처의 모델 크기에서는 데이터를 늘리는 것이 오히려 시험 성능을 해칠 수 있음을 보였다. 실무적 함의는 "모델이 보간 임계점 근처에 있으면 결과가 불안정하다"는 것이다.

## 암묵적 정규화

**왜 필요한가.** 과매개변수화 영역에서는 학습 손실 0을 달성하는 해가 무수히 많다. 손실 함수는 그중 어느 것도 선호하지 않는다. 그런데도 학습된 모델이 일관되게 잘 일반화한다면, 해를 고르는 무언가가 손실 **밖에** 있다. 그것이 학습 알고리즘(optimizer, 초기화, 구조)이 만드는 **암묵적 정규화**(implicit regularization) 또는 암묵적 편향(implicit bias)이다.

**경사하강은 최소 노름 해를 고른다.** 과소결정 최소제곱 문제 $\min_{\mathbf{w}} \frac12\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2$($\mathbf{X}$는 $N \times p$, $p > N$)에 경사하강을 적용하면 업데이트는 $-\eta\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$로, 항상 $\mathbf{X}$의 **행 공간**(7.3장) 안에 있다. 따라서 $\mathbf{w}_t - \mathbf{w}_0$는 행 공간을 벗어나지 않는다. 수렴하면 $\mathbf{w}_\infty$는 데이터를 보간하는 해 중 $\mathbf{w}_0$에 $\ell_2$ 거리로 가장 가까운 해다. $\mathbf{w}_0 = \mathbf{0}$이면 최소 노름 해 $\mathbf{X}^{+}\mathbf{y}$와 정확히 같다. 앞 절의 `pinv`는 0에서 시작한 경사하강이 가는 곳을 미리 계산한 것이다. 초기값만 바꿔 확인한다.

```python {exec="browser" session="modern"}
rng = np.random.default_rng(0)
X = rng.normal(size=(20, 100))                     # 표본 20개, 파라미터 100개, (N, p)
w_true = rng.normal(size=100) / 10
y = X @ w_true                                     # 잡음 없는 관측, (N,)
w_pinv = np.linalg.pinv(X) @ y                     # 최소 노름 보간 해
lr = 1.0 / np.linalg.norm(X, 2) ** 2               # 안정 조건 η < 2/σ_max²
for name, w in (("0에서 시작", np.zeros(100)), ("무작위 초기값", rng.normal(size=100))):
    for _ in range(3000):
        w = w - lr * X.T @ (X @ w - y)             # 0.5‖Xw - y‖²의 경사하강
    print(f"{name:8s} 학습 잔차 < 1e-8: {np.linalg.norm(X @ w - y) < 1e-8}   ‖w‖ = {np.linalg.norm(w):5.2f}   "
          f"‖w - w_pinv‖ = {np.linalg.norm(w - w_pinv):5.2f}   ‖w - w_true‖ = {np.linalg.norm(w - w_true):5.2f}")
print(f"최소 노름 해     ‖w_pinv‖ = {np.linalg.norm(w_pinv):5.2f}")
```

```output
0에서 시작   학습 잔차 < 1e-8: True   ‖w‖ =  0.45   ‖w - w_pinv‖ =  0.00   ‖w - w_true‖ =  0.90
무작위 초기값  학습 잔차 < 1e-8: True   ‖w‖ =  8.50   ‖w - w_pinv‖ =  8.49   ‖w - w_true‖ =  8.56
최소 노름 해     ‖w_pinv‖ =  0.45
```

두 실행 모두 학습 데이터를 완벽히 맞췄다. 학습 손실로는 구별할 수 없다. 그러나 0에서 시작한 해는 최소 노름 해와 같고, 무작위 초기값에서 시작한 해는 초기값의 영공간 성분(행 공간에 직교하는 성분)을 그대로 지닌 채 끝나 참 계수에서 훨씬 멀다. 경사하강은 영공간 방향으로는 한 번도 움직이지 않기 때문이다. **같은 손실, 같은 데이터, 같은 optimizer에서도 초기화가 일반화를 바꾼다.**

::quiz{id="gd-min-norm"}

**다른 암묵적 편향들.** 같은 원리가 여러 곳에서 나타난다.

- **분류와 max-margin.** 선형 분리 가능한 데이터에서 정규화 없는 로지스틱 회귀에 경사하강을 쓰면, 가중치의 크기는 끝없이 커지지만(10.1장의 실패 사례) 방향은 최대 마진(hard-margin SVM) 해로 수렴한다(Soudry 외, 2018). 수렴은 아주 느리다. 학습 오차가 0이 된 뒤에도 학습을 계속하면 일반화가 나아질 수 있는 이유 중 하나다. 이 편향은 경사하강에 특정한 것이고, Adam 같은 적응적 방법은 다른 편향을 갖는다.
- **early stopping.** 10.3장에서 본 대로 경사하강은 큰 특잇값 방향부터 맞춘다. 일찍 멈추면 작은 특잇값 방향(잡음이 증폭되는 방향)이 덜 맞춰져 ridge와 비슷한 효과를 낸다.
- **SGD의 잡음과 학습률.** 작은 batch의 잡음과 큰 학습률이 평평한 최솟값을 선호하게 만든다는 관찰과 이론이 있지만, 일반화와의 관계는 아직 논쟁 중이다(10.6장의 손실 지형).
- **구조.** CNN의 가중치 공유, 그래프 신경망의 순열 불변성(18장)은 명시적 정규화 항 없이 해의 집합 자체를 제한한다.

연구에서의 함의는 분명하다. 신경망의 일반화를 말할 때는 손실 함수뿐 아니라 **optimizer, 학습률, batch 크기, 초기화, 학습 길이**까지가 모델의 정의에 포함된다. 논문을 재현할 때 이 값들이 빠져 있으면 같은 손실로도 다른 모델이 나온다.

## 이론적 관점

**NTK와 lazy regime.** 폭이 무한히 넓은 신경망을 특정한 스케일로 초기화하면, 학습 중에 파라미터가 초기값에서 아주 조금만 움직이고 신경망이 초기값 근처의 1차 Taylor 근사(7.2장)처럼 행동한다. 이때 경사하강 학습은 고정된 커널 $K(x, x') = \nabla_\theta f(x)^\top\nabla_\theta f(x')$를 쓰는 커널 회귀(16.3장)와 같아진다. 이 커널을 **neural tangent kernel**(NTK)이라 한다(Jacot, Gabriel, Hongler, 2018). 이 관점에서 과매개변수화 신경망의 학습은 볼록 문제가 되고, 일반화는 커널 방법의 이론으로 분석된다. 앞 절의 최소 노름 회귀가 그 축소판이다.

**한계: feature learning.** 그러나 파라미터가 거의 움직이지 않는다는 것은 신경망이 **특징을 배우지 않는다**는 뜻이다. Chizat, Oyallon, Bach(2019)는 이 "lazy training"이 과매개변수화 자체가 아니라 (흔히 암묵적인) 스케일 선택의 결과이며, 실제로 잘 동작하는 신경망은 lazy regime 밖에서 특징을 학습한다는 증거를 제시했다. 사전학습한 표현을 다른 과제에 옮기는 전이학습(19.9장)은 학습된 특징이 있어야 의미가 있다. NTK는 "왜 과매개변수화 신경망을 최적화할 수 있는가"를 설명하는 데는 유용하지만, "왜 신경망이 커널 방법보다 잘하는가"를 설명하지는 못한다.

**μP의 아이디어.** 폭을 키울 때 초기화 크기와 층별 학습률을 어떻게 함께 바꾸느냐에 따라, 무한 폭의 극한이 lazy regime이 되기도 하고 특징을 학습하는 극한이 되기도 한다. Yang과 Hu(2021)는 폭이 무한대로 가도 모든 층이 특징을 학습하는 parametrization을 제안했고, 이를 maximal update parametrization(μP)이라 부른다. 이 parametrization에서는 최적 하이퍼파라미터(학습률 등)가 폭에 거의 무관해지므로, 작은 모델에서 고른 하이퍼파라미터를 큰 모델에 그대로 옮길 수 있다는 결과가 보고되었다(Yang 외, 2021, μTransfer). 실무적 의미는 "parametrization이 다르면 작은 모델의 튜닝 결과가 큰 모델로 옮겨지지 않을 수 있다"는 것이다.

:::note[심화]
세 관점을 한 줄로 요약하면 다음과 같다. NTK는 신경망을 "학습 중 고정된 커널을 쓰는 선형 모델"로 본다. feature learning 관점은 그 커널(표현) 자체가 학습 중에 데이터에 맞게 바뀐다고 본다. μP는 폭을 키울 때 두 극한 중 어느 쪽으로 갈지를 parametrization이 정한다고 본다. 이 이론들은 무한 폭 같은 극한에서 엄밀하고, 유한한 실제 모델에는 근사로만 적용된다.
:::

## 스케일링 법칙

**왜 필요한가.** 큰 모델을 학습하는 데는 막대한 계산이 든다. 어떤 크기의 모델을 얼마나 많은 데이터로 학습할지 실행 전에 정하려면, 작은 실험의 결과로 큰 실험의 결과를 **예측**할 수 있어야 한다.

**정의.** 시험 손실 $L$이 모델 크기 $N_{\text{param}}$, 데이터 크기 $D$, 계산량 $C$의 거듭제곱 함수(멱법칙)로 잘 맞는다는 경험적 관찰을 **스케일링 법칙**(scaling law)이라 한다. Kaplan 외(2020)는 언어 모델의 cross-entropy 손실이 다른 두 요인이 병목이 아닐 때 세 양 각각의 멱법칙을 따르며, 그 경향이 여러 자릿수에 걸쳐 유지된다고 보고했다. Hoffmann 외(2022)는 손실을 다음 꼴로 맞췄다.

$$
L(N_{\text{param}}, D) = E + \frac{A}{N_{\text{param}}^{\alpha}} + \frac{B}{D^{\beta}}.
$$

$E$는 데이터 자체의 엔트로피처럼 어떤 모델로도 줄일 수 없는 **바닥**(14.1장의 $H(p)$, 14.4장의 $R^\star$), 둘째·셋째 항은 모델 크기와 데이터 크기가 유한해서 생기는 초과 손실이다. 로그–로그 그림에서 바닥을 뺀 손실이 직선이 된다.

**compute-optimal 학습.** 계산량이 정해져 있을 때(트랜스포머에서 흔히 쓰는 근사는 $C \approx 6N_{\text{param}}D$ FLOP), 모델을 키울지 데이터를 늘릴지 정해야 한다. Hoffmann 외(2022)는 70M–16B 파라미터 모델 400개 이상을 학습해, 계산량이 늘 때 모델 크기와 학습 token 수를 **같은 비율로** 늘리는 것이 최적이라고 결론지었다. 그 근거로 Gopher(280B 파라미터)와 같은 계산량으로 70B 파라미터 모델 Chinchilla를 1.4T token으로 학습해 더 나은 성능을 보고했다. 이는 데이터보다 모델 크기를 더 빠르게 키우라는 Kaplan 외(2020)의 권고를 수정한 것이다. 같은 현상을 측정한 두 연구가 실험 설계(학습률 스케줄, 크기 범위)의 차이로 다른 권고를 낸 것은 스케일링 법칙 자체가 실험 설정에 의존하는 **경험적 맞춤**이라는 점을 보여 준다.

::quiz{id="compute-optimal"}

**해석의 주의점: 바닥과 외삽.** 스케일링 법칙의 가치는 외삽에 있는데, 외삽은 늘 위험하다. 가장 흔한 실수는 바닥 $E$를 무시하고 순수한 멱법칙 $L = aN^{b}$를 맞추는 것이다. 작은 범위에서는 바닥이 잘 보이지 않으므로 순수 멱법칙도 잘 맞는다. 5-최근접 이웃 회귀(잡음 표준편차 0.2)의 시험 MSE를 데이터 크기 $N$의 함수로 재고, $N \le 200$에서만 멱법칙을 맞춰 $N = 3200$까지 외삽한다.

```python {exec="browser" session="modern"}
def knn_test_mse(n_train, seed, k=5, noise=0.2):
    r = np.random.default_rng(seed)
    f = lambda x: np.sin(3 * x[:, 0]) * np.cos(3 * x[:, 1])
    x = r.uniform(0, 1, (n_train, 2)); y = f(x) + noise * r.normal(size=n_train)
    xt = r.uniform(0, 1, (300, 2)); yt = f(xt) + noise * r.normal(size=300)
    d2 = ((xt[:, None, :] - x[None, :, :]) ** 2).sum(-1)       # (300, N)
    idx = np.argpartition(d2, k, axis=1)[:, :k]                # 가까운 이웃 k개
    return np.mean((y[idx].mean(1) - yt) ** 2)

Ns = np.array([25, 50, 100, 200, 400, 800, 1600, 3200])
L = np.array([np.mean([knn_test_mse(N, s) for s in range(20)]) for N in Ns])
small = Ns <= 200                                  # 작은 범위에서만 법칙을 맞춘다
b, log_a = np.polyfit(np.log(Ns[small]), np.log(L[small]), 1)  # L = a N^b (바닥 없음)
print("    N   측정 MSE   멱법칙 외삽")
for N, l in zip(Ns, L):
    print(f"{N:5d}   {l:.4f}     {np.exp(log_a) * N**b:.4f}{'' if N <= 200 else '  ← 외삽'}")
print(f"맞춘 지수 b = {b:.3f},  잡음 분산 = {0.2**2:.4f}")
```

```output
    N   측정 MSE   멱법칙 외삽
   25   0.0817     0.0787
   50   0.0650     0.0674
  100   0.0556     0.0577
  200   0.0513     0.0495
  400   0.0490     0.0424  ← 외삽
  800   0.0494     0.0363  ← 외삽
 1600   0.0486     0.0311  ← 외삽
 3200   0.0495     0.0266  ← 외삽
맞춘 지수 b = -0.223,  잡음 분산 = 0.0400
```

맞춘 범위($N \le 200$) 안에서는 멱법칙이 측정과 잘 맞는다. 외삽은 $N = 3200$에서 0.0266을 예측하지만 실제는 0.0495로, 데이터를 16배 늘려도 거의 나아지지 않는다. 이 모델에는 바닥이 있기 때문이다. 이웃 5개를 평균하는 예측은 잡음 분산 0.04에 이웃 평균의 분산 $0.04/5$를 더한 약 0.048 아래로 내려갈 수 없다. 바닥은 데이터의 잡음($E$)에서도, 모델족의 한계(14.4장의 근사 오차)에서도 온다. 작은 범위의 데이터만으로는 바닥을 추정하기 어렵고, 바닥을 무시한 외삽은 개선을 체계적으로 과대평가한다.

::quiz{id="scaling-floor"}

스케일링 법칙을 연구 결정에 쓸 때는 다음을 확인한다.

1. **맞춘 범위가 몇 자릿수인가.** 한두 자릿수의 데이터로 맞춘 법칙을 세 자릿수 밖으로 외삽하지 않는다.
2. **바닥 항을 포함했는가.** 바닥을 포함해 맞추고, 바닥의 불확실성(부트스트랩, 12.4장)을 함께 보고한다.
3. **다른 요인이 병목이 아닌가.** 데이터 크기의 법칙을 잴 때 모델이 너무 작으면 모델이 병목이 되어 곡선이 일찍 평평해진다.
4. **지표가 바뀌지 않는가.** 손실(cross-entropy)이 매끄럽게 줄어도, 정확도처럼 문턱이 있는 지표는 갑자기 뛰는 것처럼 보일 수 있다. 손실의 법칙을 다른 지표의 예측으로 바로 옮기지 않는다.

:::note[분야 예시]
기계학습 원자간 퍼텐셜이나 분자 성질 예측에서도 학습 데이터(DFT 계산 수)에 따른 학습 곡선을 로그–로그로 그려 멱법칙으로 요약하는 관행이 있다. 이때 바닥은 참조 계산(DFT) 자체의 수치 잡음이나 모델 구조의 한계에서 온다. 작은 데이터셋에서 얻은 기울기로 "데이터를 10배 늘리면 오차가 몇 배 준다"고 예측해 계산 예산을 정한다면, 바닥 항과 맞춘 범위를 함께 보고해야 그 예측을 믿을 수 있다.
:::

## 보간과 외삽

**두 가지 의미.** "보간"이라는 말은 두 가지 뜻으로 쓰인다. 앞 절들에서는 학습 데이터를 **정확히 맞춘다**는 뜻(interpolating model)이었다. 이 절에서는 새 입력이 학습 데이터가 덮은 영역 **안에** 있다는 뜻이다. 7.2장에서 학습 데이터의 convex hull 안을 보간, 밖을 외삽이라 불렀다.

**고차원에서는 거의 모든 것이 외삽이다.** 같은 분포에서 새로 뽑은 입력이라도, 차원이 높으면 학습 데이터의 convex hull 밖에 떨어질 확률이 아주 커진다(Balestriero, Pesenti, LeCun, 2021). 표준 가우시안에서 학습 입력 500개와 새 입력 100개를 뽑고, 새 입력이 학습 데이터의 convex 결합으로 쓰일 수 있는지를 선형계획으로 판정한다.

```python {exec="browser" session="modern"}
from scipy.optimize import linprog

def in_hull(points, q):                            # q = Σ λ_i x_i, Σ λ_i = 1, λ ≥ 0 이 가능한가
    A = np.vstack([points.T, np.ones(len(points))])
    res = linprog(np.zeros(len(points)), A_eq=A, b_eq=np.append(q, 1.0), bounds=(0, None), method="highs")
    return res.status == 0

rng = np.random.default_rng(0)
print("차원 d   학습 500개의 convex hull 안에 든 새 입력의 비율")
for d in (1, 2, 4, 8, 16):
    train = rng.normal(size=(500, d))              # (N, d)
    test = rng.normal(size=(100, d))               # 같은 분포에서 뽑은 새 입력, (100, d)
    frac = np.mean([in_hull(train, q) for q in test])
    print(f"{d:5d}      {frac:.2f}")
```

```output
차원 d   학습 500개의 convex hull 안에 든 새 입력의 비율
    1      1.00
    2      0.95
    4      0.85
    8      0.39
   16      0.01
```

1차원에서는 새 입력이 거의 모두 학습 범위 안에 있지만, 16차원에서는 100개 중 1개만 convex hull 안에 있다. 분포는 바뀌지 않았는데도 그렇다. 그래서 "보간 영역에서는 믿을 수 있고 외삽 영역에서는 못 믿는다"는 기준을 고차원 입력에 그대로 쓸 수 없다. 대부분의 시험 입력이 이 정의로는 외삽이기 때문이다. 고차원에서 의미 있는 질문은 convex hull이 아니라 **분포**에 관한 것이다. 새 입력이 학습 분포에서 나왔는가, 학습 데이터의 밀도가 높은 곳에 있는가.

::quiz{id="high-dim-extrapolation"}

**분포 밖에서의 거동.** 학습 분포 밖의 입력(out-of-distribution, OOD)에서 모델이 어떻게 행동할지는 이 장의 어떤 일반화 이론도 보장하지 않는다. 14.4장의 모든 bound는 학습과 시험이 같은 분포라는 가정 위에 서 있다. 과학 문제에서는 이 가정이 깨지는 것이 오히려 기본이다. 새 재료, 새 기후 조건, 더 긴 시간, 다른 실험 장비로 예측을 옮기는 것이 연구의 목적인 경우가 많다. 분포 이동의 종류(공변량 이동, 레이블 이동, 개념 이동)와 탐지 방법은 21.4장에서, 물리 지식을 모델에 넣어 외삽을 돕는 방법은 24장에서 다룬다.

:::note[분야 예시]
배터리 수명 예측에서 실험실의 정전류 사이클 데이터로 학습한 모델을 실제 차량의 불규칙한 사용 패턴에 적용하면, 입력 특징 하나하나는 학습 범위 안에 있어도 조합은 학습 분포 밖일 수 있다. 같은 셀 화학, 같은 온도 범위라는 이유로 "보간"이라고 판단하면 위험하다. 적용 조건이 학습 분포와 어떻게 다른지를 따로 검증해야 한다(30장).
:::

## 실습

모델 크기와 데이터 크기를 바꾸어 작은 규모의 스케일링 곡선을 그리고, 외삽 예측의 신뢰도를 평가한다. 코드는 `labs/information-and-learning-theory/modern-generalization/`에 둔다.

1. **모델 크기 축.** 본문 cell 1의 최소 노름 회귀에서 $p$를 5부터 200까지 촘촘히 바꾸며 시험 위험 곡선을 그리고, 잡음 표준편차를 0, 0.1, 0.5로 바꿔 봉우리의 높이가 어떻게 변하는지 기록한다. 최적 $\lambda$의 ridge(검증 데이터로 선택)로 같은 곡선을 그려 봉우리가 사라지는지 확인한다.
2. **데이터 크기 축.** cell 3의 k-최근접 이웃 회귀(또는 random feature ridge 회귀)에서 $N$을 25부터 6400까지 바꾼다. $N \le 400$의 점만 써서 (가) 순수 멱법칙 $aN^b$, (나) 바닥이 있는 멱법칙 $aN^b + c$를 맞추고, 둘의 외삽을 실제 측정과 비교한다. (나)의 $c$는 격자 탐색으로 맞추고, 부트스트랩으로 $c$의 신뢰구간을 구한다.
3. **외삽의 신뢰도.** 맞춘 범위의 끝을 $N_{\max} \in \{100, 200, 400, 800\}$으로 바꾸며, $N = 6400$에서의 외삽 오차가 맞춘 범위에 따라 어떻게 줄어드는지 표로 만든다.
4. 결과를 "작은 실험으로 큰 실험의 결과를 예측할 때 무엇을 함께 보고해야 하는가"에 대한 한 문단으로 정리한다.

이어서 두 연습문제를 푼다.

::exercise{id="min-norm-interpolation"}

::exercise{id="power-law-floor"}

## 실패 사례

- **작은 범위에서 얻은 스케일링 법칙을 멀리 외삽하는 것.** 바닥 항을 무시하면 개선을 크게 과대평가한다(본문에서 예측 0.0266, 실제 0.0495). 맞춘 범위, 바닥의 추정과 불확실성, 다른 요인이 병목이 아니었다는 근거를 함께 보고한다.
- **보간 임계점 근처의 모델로 결론을 내리는 것.** 파라미터 수가 학습 표본 수에 가까우면 시험 오차가 크게 흔들린다. 모델 크기나 데이터 크기를 조금 바꾼 결과가 크게 다르면 이 영역을 의심하고 정규화나 크기 조정을 검토한다.
- **"과매개변수화 모델은 과적합하지 않는다"를 일반 규칙으로 쓰는 것.** 두 번째 하강은 암묵적 정규화가 문제 구조와 맞을 때의 현상이다. 본문 예처럼 과매개변수화 영역의 최소가 작은 모델보다 나쁠 수도 있다. 크기에 대한 결정은 검증 데이터로 확인한다.
- **같은 분포의 시험 성능을 분포 밖 성능의 근거로 쓰는 것.** 시험 집합이 학습 분포에서 나왔다면 그 성능은 새 조건으로의 외삽에 대해 아무것도 말하지 않는다.

## 통과 기준

- double descent와 스케일링 법칙을 자신의 실험 설계 결정(모델 크기, 데이터 수집량, 계산 예산 배분)에 연결해 한 문단으로 설명한다.
- 최소 노름 해가 보간 임계점에서 나빠지는 이유와 0에서 시작한 경사하강이 최소 노름 해로 가는 이유를 행 공간과 특잇값으로 설명한다.
- 작은 범위의 학습 곡선에서 바닥이 있는 멱법칙을 맞추고, 외삽 예측과 그 불확실성을 보고한다.
- 연습문제 두 개(`min-norm-interpolation`, `power-law-floor`)의 테스트를 통과한다.

## 참고 문헌

- Belkin, Hsu, Ma, Mandal (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. *Proceedings of the National Academy of Sciences*, 116(32), 15849–15854. — double descent 곡선의 제안
- Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever (2020). Deep double descent: Where bigger models and more data hurt. *ICLR 2020*. arXiv:1912.02292. — 신경망에서의 모델 크기·epoch 축 double descent
- Hastie, Montanari, Rosset, Tibshirani (2022). Surprises in high-dimensional ridgeless least squares interpolation. *The Annals of Statistics*, 50(2), 949–986. — 최소 노름 회귀의 위험에 대한 정확한 분석
- Soudry, Hoffer, Nacson, Gunasekar, Srebro (2018). The implicit bias of gradient descent on separable data. *Journal of Machine Learning Research*, 19(70), 1–57. — 분리 가능한 데이터에서 경사하강의 max-margin 편향
- Jacot, Gabriel, Hongler (2018). Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS 2018*. arXiv:1806.07572. — NTK
- Chizat, Oyallon, Bach (2019). On lazy training in differentiable programming. *NeurIPS 2019*. arXiv:1812.07956. — lazy training이 스케일 선택에서 온다는 분석
- Yang, Hu (2021). Tensor programs IV: Feature learning in infinite-width neural networks. *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 11727–11737. — 무한 폭에서 특징을 학습하는 parametrization(μP)
- Yang 외 (2021). Tensor programs V: Tuning large neural networks via zero-shot hyperparameter transfer. *NeurIPS 2021*. arXiv:2203.03466. — μP에서의 하이퍼파라미터 이전(μTransfer)
- Kaplan 외 (2020). Scaling laws for neural language models. arXiv:2001.08361. — 모델·데이터·계산량에 대한 멱법칙
- Hoffmann 외 (2022). Training compute-optimal large language models. *NeurIPS 2022*. arXiv:2203.15556. — compute-optimal 학습과 Chinchilla
- Balestriero, Pesenti, LeCun (2021). Learning in high dimension always amounts to extrapolation. arXiv:2110.09485. — 고차원에서 새 입력이 convex hull 밖에 놓인다는 관찰
