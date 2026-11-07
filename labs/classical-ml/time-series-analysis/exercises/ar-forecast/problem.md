# AR(p) 적합과 재귀 예측

평균이 0인 정상 시계열 $x_1, \dots, x_T$에 자기회귀 모델 AR($p$)

$$
x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + \cdots + \phi_p x_{t-p} + \varepsilon_t
$$

를 맞추고, 그 모델로 앞으로 $h$ 시점을 예측한다. 두 함수를 완성한다.

1. `fit_ar(x, p)`: 최소제곱으로 $\boldsymbol{\phi} = (\phi_1, \dots, \phi_p)$를 추정해 shape `(p,)` 배열로 반환한다. 절편은 쓰지 않는다. 회귀에 쓰는 행은 $t = p+1, \dots, T$이고, 행 $t$의 특징은 $(x_{t-1}, \dots, x_{t-p})$, 목표는 $x_t$다.
2. `forecast_ar(x, phi, h)`: 마지막 관측 $x_T$까지 보고 $\hat{x}_{T+1}, \dots, \hat{x}_{T+h}$를 shape `(h,)`로 반환한다. 미래의 잡음 기댓값은 0이므로, 아직 관측하지 않은 값 자리에는 **앞에서 만든 예측값**을 넣는다(재귀 예측).

$$
\hat{x}_{T+1} = \phi_1 x_T + \cdots + \phi_p x_{T+1-p}, \qquad
\hat{x}_{T+2} = \phi_1 \hat{x}_{T+1} + \phi_2 x_T + \cdots
$$

테스트는 시뮬레이션한 AR(2)에서 계수를 되찾는지, 손으로 계산한 재귀 예측과 같은지, AR(1)의 $h$ 시점 예측이 $\phi^h x_T$인지, 그리고 예측이 horizon이 길어질수록 평균 0으로 수렴하는지 확인한다.

힌트: lag 행렬은 `np.column_stack([x[p - k: len(x) - k] for k in range(1, p + 1)])`로 만들 수 있다.
