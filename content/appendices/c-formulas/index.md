---
id: appendices/formulas
title: "공식 모음"
title_en: "Formula Reference"
tier: reference
status: draft
prerequisites: []
---

# 공식 모음

연구 코드를 짜거나 논문을 읽다 보면 같은 공식을 몇 번이고 다시 찾게 된다. 이차형식의 gradient, 가우시안의 조건부분포, 두 가우시안 사이의 KL, Adam의 갱신식, 명시적 스킴의 안정 조건 같은 것들이다. 이 부록은 본문 장들에 흩어진 핵심 공식을 분야별로 모으고, 각 공식에 그것을 유도하거나 처음 쓰는 장을 링크한다.

공식은 **가정과 함께** 써야 맞다. 표에는 공식이 성립하는 조건을 짧게 붙였지만, 유도와 예외는 링크한 장에 있다. 기호는 [부록 A. 표기법](../a-notation/index.md)을 따른다. 벡터는 열벡터, gradient는 열벡터, Jacobian은 (출력 차원 × 입력 차원)이고, $\log$는 자연로그다.

## 학습 목표

- 자주 쓰는 공식을 이 책의 표기 규약에 맞는 형태로 찾아 쓴다
- 공식을 옮겨 올 때 layout·parameterization·로그 밑 같은 규약의 차이를 확인한다
- 공식을 유도한 장으로 돌아가 가정과 적용 조건을 확인한다

## 선형대수

| 이름 | 공식 | 조건·메모 | 장 |
|---|---|---|---|
| 내적과 노름 | $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top\mathbf{y}$, $\lVert \mathbf{x} \rVert_2 = \sqrt{\mathbf{x}^\top\mathbf{x}}$ | | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |
| 정규직교 기저의 좌표 | $\mathbf{v} = \sum_i c_i \mathbf{q}_i$, $c_i = \langle \mathbf{q}_i, \mathbf{v} \rangle$ | $\mathbf{q}_i$가 정규직교 | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |
| 직선 위로의 사영 | $\mathbf{p} = \dfrac{\mathbf{a}\mathbf{a}^\top}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{b}$ | | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |
| 최소제곱(정규방정식) | $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^\top\mathbf{b}$ | 잔차 $\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}$가 열공간과 직교. 계산은 QR·SVD로 | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |
| 직교행렬 | $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$, $\langle \mathbf{Q}\mathbf{u}, \mathbf{Q}\mathbf{v} \rangle = \langle \mathbf{u}, \mathbf{v} \rangle$ | 길이·각도 보존 | [7.4](../../07-linear-algebra/04-inner-products-and-projection/index.md) |
| 대칭행렬의 스펙트럼 분해 | $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ | $\mathbf{A}^\top = \mathbf{A}$이면 고윳값 실수, $\mathbf{Q}$ 직교 | [7.5](../../07-linear-algebra/05-eigen-and-symmetric/index.md) |
| trace·det와 고윳값 | $\operatorname{tr}(\mathbf{A}) = \sum_i \lambda_i$, $\det(\mathbf{A}) = \prod_i \lambda_i$ | 정사각행렬 | [7.5](../../07-linear-algebra/05-eigen-and-symmetric/index.md) |
| SVD | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \sum_{i=1}^r \sigma_i \mathbf{u}_i\mathbf{v}_i^\top$ | $\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$, $r = \operatorname{rank}(\mathbf{A})$ | [7.6](../../07-linear-algebra/06-svd-and-pca/index.md) |
| pseudoinverse | $\mathbf{A}^{+} = \mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top$ | $\boldsymbol{\Sigma}^{+}$는 0 아닌 특잇값만 역수 | [7.6](../../07-linear-algebra/06-svd-and-pca/index.md) |
| 최적 저랭크 근사 (Eckart–Young) | $\min_{\operatorname{rank}(\mathbf{B}) \le k} \lVert \mathbf{A} - \mathbf{B} \rVert_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}$ | 절단 SVD $\mathbf{A}_k$가 최적 | [7.6](../../07-linear-algebra/06-svd-and-pca/index.md) |
| 설명 분산 비율 | $\sum_{i \le k} \sigma_i^2 \big/ \sum_i \sigma_i^2$ | $\lVert \mathbf{A} \rVert_F^2 = \sum_i \sigma_i^2$ | [7.6](../../07-linear-algebra/06-svd-and-pca/index.md) |
| PCA와 SVD | $\mathbf{C} = \dfrac{\mathbf{X}_c^\top\mathbf{X}_c}{N - 1} = \mathbf{V}\dfrac{\boldsymbol{\Sigma}^2}{N - 1}\mathbf{V}^\top$ | $\mathbf{X}_c$는 열 평균을 뺀 $(N \times D)$ 데이터 | [7.6](../../07-linear-algebra/06-svd-and-pca/index.md) |
| trace의 성질 | $\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \operatorname{tr}(\mathbf{C}\mathbf{A}\mathbf{B})$, $\operatorname{tr}(\mathbf{A}^\top\mathbf{B}) = \sum_{ij} A_{ij}B_{ij}$ | shape이 맞을 때 순환 가능 | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| Woodbury 항등식 | $(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}\mathbf{A}^{-1}$ | 저랭크 갱신의 역. 공분산 형태 ↔ 정밀도 형태 변환에 쓴다 | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |

## 미분과 행렬 미분

이 책은 numerator layout을 쓴다. 스칼라 $L$을 행렬 $\mathbf{W}$로 미분한 gradient는 $\mathbf{W}$와 같은 shape이다. 미분형 $dL = \operatorname{tr}(\mathbf{G}^\top d\mathbf{X})$로 정리되면 $\nabla_{\mathbf{X}} L = \mathbf{G}$로 읽는다([8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md)).

| 이름 | 공식 | 조건·메모 | 장 |
|---|---|---|---|
| 1차 근사 | $f(\mathbf{a} + \mathbf{h}) \approx f(\mathbf{a}) + \mathbf{J}_f(\mathbf{a})\,\mathbf{h}$ | $\mathbf{J}_f \in \mathbb{R}^{m \times n}$ | [8.1](../../08-calculus-and-autodiff/01-multivariable-calculus/index.md) |
| 방향미분 | $D_{\mathbf{u}} f(\mathbf{a}) = \nabla f(\mathbf{a})^\top \mathbf{u}$ | $f$가 미분 가능 | [8.1](../../08-calculus-and-autodiff/01-multivariable-calculus/index.md) |
| 2차 Taylor 근사 | $f(\mathbf{x} + \mathbf{d}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top\mathbf{d} + \tfrac{1}{2}\mathbf{d}^\top\nabla^2 f(\mathbf{x})\,\mathbf{d}$ | Newton 방법의 모델 | [10.5](../../10-optimization/05-second-order-methods/index.md) |
| 연쇄법칙 | $\mathbf{J}_{f \circ g}(\mathbf{x}) = \mathbf{J}_f(g(\mathbf{x}))\,\mathbf{J}_g(\mathbf{x})$ | 역전파는 이것을 출력 쪽부터 $\mathbf{v}^\top\mathbf{J}$로 계산 | [8.1](../../08-calculus-and-autodiff/01-multivariable-calculus/index.md), [8.4](../../08-calculus-and-autodiff/04-automatic-differentiation/index.md) |
| 미분형 규칙 | $d(\mathbf{X}\mathbf{Y}) = (d\mathbf{X})\mathbf{Y} + \mathbf{X}\,d\mathbf{Y}$, $d(\mathbf{X}^\top) = (d\mathbf{X})^\top$ | 곱의 순서를 지킨다 | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| 선형 함수 | $\nabla_{\mathbf{x}}\, \mathbf{a}^\top\mathbf{x} = \mathbf{a}$ | | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| 이차형식 | $\nabla_{\mathbf{x}}\, \mathbf{x}^\top\mathbf{A}\mathbf{x} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ | $\mathbf{A}$가 대칭이면 $2\mathbf{A}\mathbf{x}$ | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| 최소제곱 | $\nabla_{\mathbf{x}}\, \lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert^2 = 2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$ | | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| 쌍선형형식 | $\nabla_{\mathbf{X}}\, \mathbf{a}^\top\mathbf{X}\mathbf{b} = \mathbf{a}\mathbf{b}^\top$ | | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| trace | $\nabla_{\mathbf{X}} \operatorname{tr}(\mathbf{A}\mathbf{X}) = \mathbf{A}^\top$, $\nabla_{\mathbf{X}} \operatorname{tr}(\mathbf{X}^\top\mathbf{A}\mathbf{X}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{X}$ | | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| 역행렬 | $d(\mathbf{X}^{-1}) = -\mathbf{X}^{-1}(d\mathbf{X})\,\mathbf{X}^{-1}$ | $\nabla_{\mathbf{X}} \operatorname{tr}(\mathbf{A}\mathbf{X}^{-1}) = -\mathbf{X}^{-\top}\mathbf{A}^\top\mathbf{X}^{-\top}$ | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| log det (Jacobi 공식) | $d \log\det\mathbf{X} = \operatorname{tr}(\mathbf{X}^{-1}d\mathbf{X})$, $\nabla_{\mathbf{X}} \log\det\mathbf{X} = \mathbf{X}^{-\top}$ | $\det\mathbf{X} > 0$ | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| 선형층의 가중치 gradient | $L = \tfrac{1}{2}\lVert \mathbf{W}\mathbf{x} - \mathbf{y} \rVert^2 \Rightarrow \nabla_{\mathbf{W}} L = \mathbf{r}\mathbf{x}^\top$ | $\mathbf{r} = \mathbf{W}\mathbf{x} - \mathbf{y}$. 출력 쪽 gradient와 입력의 외적 | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| softmax cross-entropy | $\nabla_{\mathbf{z}} L = \mathbf{p} - \mathbf{y}$ | $p_k = e^{z_k}/\sum_j e^{z_j}$, $L = -\sum_k y_k \log p_k$, $\mathbf{y}$ one-hot | [8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md) |
| 변수변환 | $\int_{\mathcal{X}} f(\mathbf{x})\,d\mathbf{x} = \int_{\mathcal{U}} f(g(\mathbf{u}))\,\lvert \det \mathbf{J}_g(\mathbf{u}) \rvert\,d\mathbf{u}$ | $g$가 가역이고 연속 미분 가능 | [8.3](../../08-calculus-and-autodiff/03-integration/index.md) |
| 밀도의 변수변환 | $p_X(\mathbf{x}) = p_U(g^{-1}(\mathbf{x}))\,\lvert \det \mathbf{J}_{g^{-1}}(\mathbf{x}) \rvert$ | normalizing flow의 바탕 | [8.3](../../08-calculus-and-autodiff/03-integration/index.md) |
| 가우시안 적분 | $\int_{\mathbb{R}} e^{-x^2/2}\,dx = \sqrt{2\pi}$, $\int_{\mathbb{R}^d} e^{-\frac{1}{2}\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}}\,d\mathbf{x} = (2\pi)^{d/2}\det(\boldsymbol{\Sigma})^{1/2}$ | $\boldsymbol{\Sigma}$ 양의 정부호 | [8.3](../../08-calculus-and-autodiff/03-integration/index.md) |

::quiz{id="quadratic-form-gradient"}

## 확률

| 이름 | 공식 | 조건·메모 | 장 |
|---|---|---|---|
| 주변화 | $p(x) = \sum_y p(x, y)$ (연속이면 $\int p(x, y)\,dy$) | | [11.3](../../11-probability/03-conditional-probability/index.md) |
| 조건부분포·곱의 법칙 | $p(y \mid x) = \dfrac{p(x, y)}{p(x)}$, $p(x, y) = p(y \mid x)\,p(x)$ | $p(x) > 0$ | [11.3](../../11-probability/03-conditional-probability/index.md) |
| Bayes 정리 | $p(\theta \mid \mathcal{D}) = \dfrac{p(\mathcal{D} \mid \theta)\,p(\theta)}{p(\mathcal{D})} \propto p(\mathcal{D} \mid \theta)\,p(\theta)$ | | [11.3](../../11-probability/03-conditional-probability/index.md) |
| 독립 | $X \perp Y \iff p(x, y) = p(x)\,p(y)$ | 모든 $x, y$에 대해 | [11.3](../../11-probability/03-conditional-probability/index.md) |
| 기댓값의 선형성 | $\mathbb{E}[aX + bY + c] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] + c$ | 독립이 아니어도 성립 | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| 분산 | $\operatorname{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| 합의 분산 | $\operatorname{Var}[aX + bY] = a^2\operatorname{Var}[X] + b^2\operatorname{Var}[Y] + 2ab\operatorname{Cov}[X, Y]$ | 독립이면 공분산 항이 0 | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| 반복 기댓값 | $\mathbb{E}\big[\mathbb{E}[Y \mid X]\big] = \mathbb{E}[Y]$ | | [11.3](../../11-probability/03-conditional-probability/index.md) |
| 전체 분산의 법칙 | $\operatorname{Var}[Y] = \mathbb{E}\big[\operatorname{Var}[Y \mid X]\big] + \operatorname{Var}\big[\mathbb{E}[Y \mid X]\big]$ | 앙상블 불확실성 분해에 쓰인다 | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| Markov 부등식 | $\Pr(X \ge a) \le \mathbb{E}[X]/a$ | $X \ge 0$, $a > 0$ | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| Chebyshev 부등식 | $\Pr(\lvert X - \mu \rvert \ge k\sigma) \le 1/k^2$ | 분산이 유한 | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| Hoeffding 부등식 | $\Pr(\lvert \bar{X}_n - \mu \rvert \ge t) \le 2\exp\!\big(-2nt^2/(b - a)^2\big)$ | 독립, 각 $X_i \in [a, b]$ | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| 중심극한정리 | $\sqrt{n}\,(\bar{X}_n - \mu)/\sigma \xrightarrow{d} \mathcal{N}(0, 1)$ | i.i.d., 분산 유한 | [11.5](../../11-probability/05-expectation-and-limits/index.md) |
| Monte Carlo 추정 | $\hat{I}_N = \tfrac{1}{N}\sum_{i=1}^N f(x_i)$, $\mathrm{SE}(\hat{I}_N) = \sigma_f/\sqrt{N}$ | $x_i \sim p$ 독립 | [11.6](../../11-probability/06-sampling/index.md) |
| 역변환 표집 | $U \sim \operatorname{Unif}(0, 1) \Rightarrow F^{-1}(U) \sim F$ | | [11.6](../../11-probability/06-sampling/index.md) |
| 중요도 표집 | $\mathbb{E}_p[f(X)] = \mathbb{E}_q[f(X)\,w(X)]$, $w(x) = p(x)/q(x)$ | $f(x)p(x) \neq 0$인 곳에서 $q(x) > 0$ | [11.6](../../11-probability/06-sampling/index.md) |
| 자기정규화 중요도 표집 | $\hat{I}_{\mathrm{SN}} = \sum_i w_i f(x_i) \big/ \sum_i w_i$ | 정규화 상수를 몰라도 된다. 편향 있음 | [11.6](../../11-probability/06-sampling/index.md) |
| 유효 표본 크기 | $\mathrm{ESS} = \big(\sum_i w_i\big)^2 \big/ \sum_i w_i^2$ | 가중치가 고를수록 $N$에 가깝다 | [11.6](../../11-probability/06-sampling/index.md) |

## 가우시안 항등식

$D$차원 가우시안 $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$의 밀도는

$$
p(\mathbf{x}) = \frac{1}{(2\pi)^{D/2}\,\det(\boldsymbol{\Sigma})^{1/2}} \exp\!\Big(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\Big)
$$

이다([11.4](../../11-probability/04-multivariate-gaussian/index.md)). 벡터를 관측하지 않은 부분 $\mathbf{x}_a$와 관측한 부분 $\mathbf{x}_b$로 나누고 평균과 공분산을 같은 블록으로 나누면

$$
\mathbf{x}_a \mid \mathbf{x}_b \sim \mathcal{N}\Big(\boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\mathbf{x}_b - \boldsymbol{\mu}_b),\ \ \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}\Big)
$$

이다. 조건부 평균은 관측값에 선형이고, 조건부 공분산은 관측값 $\mathbf{x}_b$에 의존하지 않는다.

| 이름 | 공식 | 조건·메모 | 장 |
|---|---|---|---|
| Mahalanobis 거리 | $d_M(\mathbf{x})^2 = (\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$ | 밀도의 등고면 | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| 선형변환 | $\mathbf{A}\mathbf{x} + \mathbf{b} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b},\ \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$ | | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| 표집 | $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$, $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ (Cholesky) | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| 주변분포 | $\mathbf{x}_a \sim \mathcal{N}(\boldsymbol{\mu}_a, \boldsymbol{\Sigma}_{aa})$ | 해당 블록만 읽는다 | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| 조건부분포(정밀도 형태) | $\mathbf{x}_a \mid \mathbf{x}_b \sim \mathcal{N}\big(\boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{aa}^{-1}\boldsymbol{\Lambda}_{ab}(\mathbf{x}_b - \boldsymbol{\mu}_b),\ \boldsymbol{\Lambda}_{aa}^{-1}\big)$ | $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| 조건부 독립과 정밀도 | $\Lambda_{ij} = 0 \iff x_i \perp x_j \mid \mathbf{x}_{\setminus ij}$ | Gaussian graphical model | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| 선형 가우시안 모델의 사후분포 | $\mathbf{S} = \mathbf{A}\mathbf{P}_0\mathbf{A}^\top + \mathbf{R}$, $\mathbf{K} = \mathbf{P}_0\mathbf{A}^\top\mathbf{S}^{-1}$, $\mathbf{x} \mid \mathbf{y} \sim \mathcal{N}\big(\mathbf{m}_0 + \mathbf{K}(\mathbf{y} - \mathbf{A}\mathbf{m}_0),\ \mathbf{P}_0 - \mathbf{K}\mathbf{A}\mathbf{P}_0\big)$ | $\mathbf{x} \sim \mathcal{N}(\mathbf{m}_0, \mathbf{P}_0)$, $\mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$. Kalman 갱신과 같다 | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| 같은 사후분포(정밀도 형태) | $\mathbf{P}^{-1} = \mathbf{P}_0^{-1} + \mathbf{A}^\top\mathbf{R}^{-1}\mathbf{A}$, $\mathbf{m} = \mathbf{P}\big(\mathbf{P}_0^{-1}\mathbf{m}_0 + \mathbf{A}^\top\mathbf{R}^{-1}\mathbf{y}\big)$ | Woodbury로 위 식과 오간다 | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |
| 관측의 주변분포 | $\mathbf{y} \sim \mathcal{N}(\mathbf{A}\mathbf{m}_0,\ \mathbf{A}\mathbf{P}_0\mathbf{A}^\top + \mathbf{R})$ | evidence. 모델 비교에 쓴다 | [11.4](../../11-probability/04-multivariate-gaussian/index.md) |

::quiz{id="gaussian-conditional-variance"}

## 분포 표

**이산 분포**([11.2](../../11-probability/02-distributions/index.md)). `scipy.stats` 열은 집필 시점(SciPy 1.14) 예시이며, 쓰는 버전의 문서에서 인자 이름을 확인한다.

| 분포 | 값 | PMF $p(k)$ | 평균 | 분산 | `scipy.stats` |
|---|---|---|---|---|---|
| Bernoulli($\pi$) | $\{0, 1\}$ | $\pi^k(1-\pi)^{1-k}$ | $\pi$ | $\pi(1-\pi)$ | `bernoulli(pi)` |
| binomial($n, \pi$) | $\{0, \dots, n\}$ | $\binom{n}{k}\pi^k(1-\pi)^{n-k}$ | $n\pi$ | $n\pi(1-\pi)$ | `binom(n, pi)` |
| categorical($\boldsymbol{\pi}$) | $\{1, \dots, K\}$ | $\pi_k$ | — | — | (`multinomial(1, pi)`) |
| Poisson($\lambda$) | $\{0, 1, 2, \dots\}$ | $e^{-\lambda}\lambda^k/k!$ | $\lambda$ | $\lambda$ | `poisson(lam)` |
| geometric($\pi$) | $\{1, 2, \dots\}$ | $(1-\pi)^{k-1}\pi$ | $1/\pi$ | $(1-\pi)/\pi^2$ | `geom(pi)` |

**연속 분포**([11.2](../../11-probability/02-distributions/index.md)). 밀도는 정규화 상수를 생략했다. 지수분포와 gamma의 $\lambda$, $\beta$는 **rate**이고, SciPy는 **scale** $= 1/\text{rate}$를 받는다.

| 분포 | 값 | 밀도 $p(x)$ (상수 생략) | 평균 | 분산 | `scipy.stats` |
|---|---|---|---|---|---|
| uniform($a, b$) | $[a, b]$ | $1$ | $(a+b)/2$ | $(b-a)^2/12$ | `uniform(loc=a, scale=b-a)` |
| Gaussian($\mu, \sigma^2$) | $\mathbb{R}$ | $e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | `norm(mu, sigma)` |
| exponential($\lambda$) | $[0, \infty)$ | $e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | `expon(scale=1/lam)` |
| gamma($\alpha, \beta$) | $(0, \infty)$ | $x^{\alpha-1}e^{-\beta x}$ | $\alpha/\beta$ | $\alpha/\beta^2$ | `gamma(a=alpha, scale=1/beta)` |
| beta($a, b$) | $[0, 1]$ | $x^{a-1}(1-x)^{b-1}$ | $a/(a+b)$ | $\frac{ab}{(a+b)^2(a+b+1)}$ | `beta(a, b)` |
| Student-t($\nu$) | $\mathbb{R}$ | $(1 + x^2/\nu)^{-(\nu+1)/2}$ | $0$ ($\nu > 1$) | $\nu/(\nu-2)$ ($\nu > 2$) | `t(df=nu)` |
| log-normal($\mu, \sigma^2$) | $(0, \infty)$ | $\frac{1}{x}e^{-(\log x-\mu)^2/2\sigma^2}$ | $e^{\mu+\sigma^2/2}$ | $(e^{\sigma^2}-1)e^{2\mu+\sigma^2}$ | `lognorm(s=sigma, scale=exp(mu))` |
| Laplace($\mu, b$) | $\mathbb{R}$ | $e^{-\lvert x-\mu\rvert/b}$ | $\mu$ | $2b^2$ | `laplace(mu, b)` |

**지수족**([11.2](../../11-probability/02-distributions/index.md), [12.2](../../12-statistical-inference/02-likelihood/index.md)). $p(x \mid \boldsymbol{\eta}) = h(x)\exp\big(\boldsymbol{\eta}^\top T(x) - A(\boldsymbol{\eta})\big)$이면 $\nabla A(\boldsymbol{\eta}) = \mathbb{E}[T(X)]$, $\nabla^2 A(\boldsymbol{\eta}) = \operatorname{Cov}[T(X)]$이고, 최대우도 조건은 $\nabla A(\hat{\boldsymbol{\eta}}) = \frac{1}{N}\sum_i T(x_i)$(모멘트 맞추기)다.

| 분포 | $T(x)$ | 자연 파라미터 $\eta$ | $A(\eta)$ |
|---|---|---|---|
| Bernoulli($\pi$) | $x$ | $\log\frac{\pi}{1-\pi}$ (logit) | $\log(1 + e^\eta)$ |
| categorical($\boldsymbol{\pi}$) | one-hot 벡터 | $\log \pi_k$ | $\log\sum_k e^{\eta_k}$ |
| Poisson($\lambda$) | $x$ | $\log \lambda$ | $e^\eta$ |
| Gaussian($\mu, \sigma^2$) | $(x, x^2)$ | $(\mu/\sigma^2, -1/2\sigma^2)$ | $-\frac{\eta_1^2}{4\eta_2} - \frac{1}{2}\log(-2\eta_2)$ |

**켤레 사전분포**([13.2](../../13-bayesian-inference/02-conjugate-models/index.md)). 사후분포가 사전분포와 같은 분포족에 머무는 쌍이다.

| 우도 | 사전분포 | 사후분포 |
|---|---|---|
| Bernoulli·binomial ($n$번 중 $h$번 성공) | $\mathrm{Beta}(a, b)$ | $\mathrm{Beta}(a + h,\ b + n - h)$ |
| Poisson (노출 $t_i$ 동안 $y_i$건) | $\mathrm{Gamma}(a, b)$ (shape, rate) | $\mathrm{Gamma}\big(a + \sum_i y_i,\ b + \sum_i t_i\big)$ |
| Gaussian (분산 $\sigma^2$ 앎, $n$개, 평균 $\bar{y}$) | $\mathcal{N}(\mu_0, \tau_0^2)$ | $\mathcal{N}(\mu_n, \tau_n^2)$, $\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}$, $\mu_n = \tau_n^2\big(\frac{\mu_0}{\tau_0^2} + \frac{n\bar{y}}{\sigma^2}\big)$ |
| 선형회귀 $\mathbf{y} \mid \mathbf{w} \sim \mathcal{N}(\boldsymbol{\Phi}\mathbf{w}, \sigma^2\mathbf{I})$ | $\mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})$ | $\mathcal{N}(\mathbf{m}, \mathbf{S})$, $\mathbf{S} = \big(\alpha\mathbf{I} + \sigma^{-2}\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\big)^{-1}$, $\mathbf{m} = \sigma^{-2}\mathbf{S}\boldsymbol{\Phi}^\top\mathbf{y}$ |

베이지안 선형회귀의 예측분포는 $p(y_\ast \mid \mathbf{x}_\ast, \mathbf{y}) = \mathcal{N}\big(\boldsymbol{\phi}_\ast^\top\mathbf{m},\ \sigma^2 + \boldsymbol{\phi}_\ast^\top\mathbf{S}\boldsymbol{\phi}_\ast\big)$이다. 첫 항이 관측 잡음(aleatoric), 둘째 항이 가중치 불확실성(epistemic)이다. 주변우도는 $\mathbf{y} \sim \mathcal{N}(\mathbf{0},\ \sigma^2\mathbf{I} + \alpha^{-1}\boldsymbol{\Phi}\boldsymbol{\Phi}^\top)$이다.

## 통계적 추론

| 이름 | 공식 | 조건·메모 | 장 |
|---|---|---|---|
| 표본 평균의 표준오차 | $\operatorname{Var}[\bar{X}] = \sigma^2/n$, $\mathrm{SE} = \sigma/\sqrt{n}$ | 독립 | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| 편향과 MSE | $\mathrm{MSE}(\hat\theta) = \mathbb{E}[(\hat\theta - \theta)^2] = \operatorname{bias}(\hat\theta)^2 + \operatorname{Var}(\hat\theta)$ | $\operatorname{bias}(\hat\theta) = \mathbb{E}[\hat\theta] - \theta$ | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| 표본 분산 | $s^2 = \dfrac{1}{n-1}\sum_i (X_i - \bar{X})^2$ | $n-1$로 나누면 불편. `np.var`의 기본은 $n$ (`ddof=0`) | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| Fisher 정보 | $I(\theta) = \mathbb{E}\big[(\partial_\theta \log p(X \mid \theta))^2\big] = -\mathbb{E}\big[\partial_\theta^2 \log p(X \mid \theta)\big]$ | 정칙 조건 | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| Cramér–Rao 하한 | $\operatorname{Var}(\hat\theta) \ge 1/(n\,I(\theta))$ | 불편추정량, i.i.d. | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| 점근 정규성 | $\sqrt{n}\,(\hat\theta_n - \theta) \xrightarrow{d} \mathcal{N}(0, v)$ | MLE는 정칙 조건에서 $v = 1/I(\theta)$ | [12.1](../../12-statistical-inference/01-estimators/index.md) |
| 최대우도 | $\hat\theta_{\text{MLE}} = \arg\max_\theta \sum_i \log p(x_i \mid \theta)$ | 독립 관측 | [12.2](../../12-statistical-inference/02-likelihood/index.md) |
| MAP | $\hat\theta_{\text{MAP}} = \arg\min_\theta \big(-\log p(\mathcal{D} \mid \theta) - \log p(\theta)\big)$ | 가우시안 사전분포는 L2 정규화와 같다 | [12.2](../../12-statistical-inference/02-likelihood/index.md) |
| 가우시안 NLL | $-\log \mathcal{N}(y; \mu, \sigma^2) = \tfrac{1}{2}\log(2\pi\sigma^2) + \dfrac{(y - \mu)^2}{2\sigma^2}$ | $\sigma$ 고정이면 MSE와 같은 최소점 | [12.3](../../12-statistical-inference/03-distributional-formulation/index.md) |
| 신뢰구간의 정의 | $\Pr_\theta[L(X) \le \theta \le U(X)] \ge 1 - \alpha$ | 모든 $\theta$에 대해. 구간이 확률변수다 | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| $t$ 통계량 | $T = \dfrac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}$ | 정규 모집단 | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| 검정력과 표본 수 | 검정력 $\approx \Phi\big(d\sqrt{n/2} - z_{1-\alpha/2}\big)$, $n \approx 2(z_{1-\alpha/2} + z_{1-\beta})^2/d^2$ | 두 집단, 집단당 $n$, $d = (\mu_1 - \mu_0)/\sigma$ | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| 가족별 오류율 | $\mathrm{FWER} = 1 - (1 - \alpha)^m$ | 독립인 $m$개 검정. Bonferroni는 각 검정을 $\alpha/m$으로 | [12.4](../../12-statistical-inference/04-intervals-and-testing/index.md) |
| 비선형 최소제곱 추정의 공분산 | $\operatorname{Cov}(\hat\theta) \approx \sigma^2(\mathbf{J}^\top\mathbf{J})^{-1}$ | 잡음 정규·등분산, 해 근처에서 거의 선형 | [10.5](../../10-optimization/05-second-order-methods/index.md) |

## 최적화

| 이름 | 공식 | 조건·메모 | 장 |
|---|---|---|---|
| 경사하강 | $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\nabla f(\mathbf{x}_k)$ | | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| 이차 함수에서의 안정 조건 | $0 < \eta < 2/L$, $L = \lambda_{\max}(\mathbf{H})$ | 넘으면 가장 가파른 방향으로 발산 | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| 수렴 속도와 조건수 | 축소율 $\dfrac{\kappa - 1}{\kappa + 1}$, $\kappa = L/\mu$ | 강볼록 이차 함수, $\eta = 2/(\mu + L)$ | [10.3](../../10-optimization/03-first-order-methods/index.md), [9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md) |
| 하강 보조정리 | $f(\mathbf{x}_{k+1}) \le f(\mathbf{x}_k) - \tfrac{\eta}{2}\lVert \nabla f(\mathbf{x}_k) \rVert^2$ | $L$-smooth, $\eta \le 1/L$ | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| Armijo 조건 | $f(\mathbf{x} - t\nabla f(\mathbf{x})) \le f(\mathbf{x}) - c\,t\,\lVert \nabla f(\mathbf{x}) \rVert^2$ | backtracking line search, $0 < c < 1$ | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| SGD | $\theta_{k+1} = \theta_k - \eta_k\,\tfrac{1}{B}\sum_{i \in \mathcal{B}_k} \nabla \ell_i(\theta_k)$ | mini-batch 크기 $B$ | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| Robbins–Monro 조건 | $\sum_k \eta_k = \infty$, $\sum_k \eta_k^2 < \infty$ | 학습률 감쇠의 근거 | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| heavy ball momentum | $\mathbf{v}_{k+1} = \beta\mathbf{v}_k - \eta\nabla f(\mathbf{x}_k)$, $\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}$ | Nesterov는 gradient를 $\mathbf{x}_k + \beta\mathbf{v}_k$에서 계산 | [10.3](../../10-optimization/03-first-order-methods/index.md) |
| Newton 방법 | $\nabla^2 f(\mathbf{x}_k)\,\mathbf{d}_k = -\nabla f(\mathbf{x}_k)$, $\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{d}_k$ | 해 근처에서 이차 수렴 | [10.5](../../10-optimization/05-second-order-methods/index.md) |
| Gauss–Newton | $\mathbf{J}^\top\mathbf{J}\,\boldsymbol{\delta} = -\mathbf{J}^\top\mathbf{r}$ | $F = \tfrac{1}{2}\lVert \mathbf{r}(\theta) \rVert^2$, $\mathbf{J} = \partial\mathbf{r}/\partial\theta$ | [10.5](../../10-optimization/05-second-order-methods/index.md) |
| Levenberg–Marquardt | $(\mathbf{J}^\top\mathbf{J} + \lambda\mathbf{I})\,\boldsymbol{\delta} = -\mathbf{J}^\top\mathbf{r}$ | $\lambda$가 크면 경사하강, 작으면 Gauss–Newton | [10.5](../../10-optimization/05-second-order-methods/index.md) |
| Lagrangian | $L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})$ | 제약 $g_i \le 0$, $h_j = 0$ | [10.2](../../10-optimization/02-constrained-optimization/index.md) |
| KKT 조건 | $\nabla_{\mathbf{x}} L = \mathbf{0}$, $g_i \le 0$, $h_j = 0$, $\lambda_i \ge 0$, $\lambda_i g_i = 0$ | 정상성·원 실행 가능성·쌍대 실행 가능성·상보 여유 | [10.2](../../10-optimization/02-constrained-optimization/index.md) |
| 쌍대 함수 | $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \min_{\mathbf{x}} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \le f^\star$ | $\boldsymbol{\lambda} \ge \mathbf{0}$이면 항상 하한(약한 쌍대성) | [10.2](../../10-optimization/02-constrained-optimization/index.md) |
| 승수의 민감도 | $df^\star/db = -\nu$ | 제약 $c(\mathbf{x}) - b = 0$에 승수 $\nu$를 붙여 Lagrangian에 더할 때 | [10.2](../../10-optimization/02-constrained-optimization/index.md) |
| 투영 경사하강 | $\mathbf{x}_{k+1} = \Pi_{\mathcal{C}}\big(\mathbf{x}_k - \eta\nabla f(\mathbf{x}_k)\big)$ | 볼록 집합 $\mathcal{C}$ 위로의 사영 | [10.2](../../10-optimization/02-constrained-optimization/index.md) |
| 근위 연산 | $\operatorname{prox}_{tR}(\mathbf{v}) = \arg\min_{\mathbf{x}} \big(R(\mathbf{x}) + \tfrac{1}{2t}\lVert \mathbf{x} - \mathbf{v} \rVert^2\big)$ | L1이면 soft thresholding | [10.2](../../10-optimization/02-constrained-optimization/index.md) |

**Adam과 AdamW**([10.4](../../10-optimization/04-adaptive-methods/index.md)). gradient $\mathbf{g}_t$의 1차·2차 모멘트를 지수이동평균으로 추적하고 초기 편향을 보정한다. 제곱과 나눗셈은 성분별이다.

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\mathbf{m}_{t-1} + (1 - \beta_1)\,\mathbf{g}_t, & \mathbf{v}_t &= \beta_2\mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2, \\
\hat{\mathbf{m}}_t &= \mathbf{m}_t / (1 - \beta_1^t), & \hat{\mathbf{v}}_t &= \mathbf{v}_t / (1 - \beta_2^t), \\
\mathbf{w}_t &= \mathbf{w}_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \eta\lambda\,\mathbf{w}_{t-1}. & &
\end{aligned}
$$

마지막 항($\eta\lambda\mathbf{w}_{t-1}$)이 AdamW의 decoupled weight decay다. 이 항을 빼면 Adam이다. Adam에 L2 항 $\tfrac{\lambda}{2}\lVert\mathbf{w}\rVert^2$을 손실에 더하는 것은 이것과 같지 않다(그 gradient가 적응적 분모로 나뉘기 때문이다).

## 정보이론

| 이름 | 공식 | 조건·메모 | 장 |
|---|---|---|---|
| 엔트로피 | $H(p) = -\sum_x p(x)\log p(x) = \mathbb{E}_{x \sim p}[-\log p(x)]$ | 자연로그면 nat, $\log_2$면 bit. $1\ \text{bit} = \log 2\ \text{nat}$ | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| 미분 엔트로피 | $h(p) = -\int p(x)\log p(x)\,dx$ | 음수일 수 있다 | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| 교차엔트로피 | $H(p, q) = \mathbb{E}_{x \sim p}[-\log q(x)] = H(p) + \mathrm{KL}(p \,\Vert\, q)$ | $p$에 대해 최소화하면 KL 최소화와 같다 | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| KL divergence | $\mathrm{KL}(p \,\Vert\, q) = \sum_x p(x)\log\dfrac{p(x)}{q(x)} \ge 0$ | 비대칭. $p > 0$인 곳에서 $q = 0$이면 $\infty$. 14.1장은 $D_{\mathrm{KL}}$로 쓴다 | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| Jensen–Shannon | $\mathrm{JS}(p, q) = \tfrac{1}{2}\mathrm{KL}(p \,\Vert\, m) + \tfrac{1}{2}\mathrm{KL}(q \,\Vert\, m)$, $m = \tfrac{1}{2}(p + q)$ | 대칭, $[0, \log 2]$ | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| $f$-divergence | $D_f(p \,\Vert\, q) = \mathbb{E}_{x \sim q}\big[f\big(p(x)/q(x)\big)\big]$ | $f$ 볼록, $f(1) = 0$. $f(t) = t\log t$이면 KL | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| 상호정보량 | $I(X; Y) = \mathrm{KL}\big(p(x, y) \,\Vert\, p(x)p(y)\big) = H(X) - H(X \mid Y)$ | $= H(X) + H(Y) - H(X, Y)$ | [14.2](../../14-information-and-learning-theory/02-mutual-information/index.md) |
| 데이터 처리 부등식 | $I(X; Z) \le I(X; Y)$ | $Z$가 $Y$만 보고 만들어질 때($X \to Y \to Z$) | [14.2](../../14-information-and-learning-theory/02-mutual-information/index.md) |
| 가우시안 엔트로피 | $h\big(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})\big) = \tfrac{1}{2}\log\big((2\pi e)^D \det\boldsymbol{\Sigma}\big)$ | 같은 공분산의 분포 중 최대 | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| 일변수 가우시안 KL | $\mathrm{KL}\big(\mathcal{N}(\mu_1, \sigma_1^2) \,\Vert\, \mathcal{N}(\mu_2, \sigma_2^2)\big) = \log\dfrac{\sigma_2}{\sigma_1} + \dfrac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \dfrac{1}{2}$ | | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| 다변량 가우시안 KL | $\tfrac{1}{2}\Big[\operatorname{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) - D + \log\dfrac{\det\boldsymbol{\Sigma}_2}{\det\boldsymbol{\Sigma}_1}\Big]$ | $\mathrm{KL}\big(\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) \,\Vert\, \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)\big)$, $D$차원 | [14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md) |
| VAE의 KL 항 | $\mathrm{KL}\big(\mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}\mathbf{s}^2) \,\Vert\, \mathcal{N}(\mathbf{0}, \mathbf{I})\big) = \tfrac{1}{2}\sum_j \big(\mu_j^2 + s_j^2 - 1 - \log s_j^2\big)$ | 위 식의 특수한 경우 | [19.3](../../19-generative-and-foundation-models/03-vae/index.md) |
| ELBO | $\log p(\mathcal{D}) = \underbrace{\mathbb{E}_q[\log p(\mathcal{D}, \theta) - \log q(\theta)]}_{\text{ELBO}} + \mathrm{KL}\big(q(\theta) \,\Vert\, p(\theta \mid \mathcal{D})\big)$ | KL $\ge 0$이므로 ELBO는 하한 | [13.6](../../13-bayesian-inference/06-variational-inference/index.md) |

## 수치해석

| 이름 | 공식 | 조건·메모 | 장 |
|---|---|---|---|
| 부동소수점 값 | $x = (-1)^s \times (1.f)_2 \times 2^{\,e - \text{bias}}$ | 정규수 | [9.1](../../09-numerical-methods/01-floating-point/index.md) |
| 반올림 모델 | $\mathrm{fl}(x) = x(1 + \delta)$, $\lvert\delta\rvert \le u = \varepsilon/2$ | float64 $\varepsilon \approx 2.2 \times 10^{-16}$, float32 $\varepsilon \approx 1.2 \times 10^{-7}$ | [9.1](../../09-numerical-methods/01-floating-point/index.md) |
| 뺄셈의 상대 오차 | $\dfrac{\lvert a\delta_a - b\delta_b \rvert}{\lvert a - b \rvert} \le \dfrac{\lvert a\rvert + \lvert b\rvert}{\lvert a - b\rvert}\,u$ | $a \approx b$이면 증폭(파국적 상쇄) | [9.1](../../09-numerical-methods/01-floating-point/index.md) |
| log-sum-exp | $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$, $m = \max_j z_j$ | overflow·underflow 방지 | [9.1](../../09-numerical-methods/01-floating-point/index.md) |
| 선형계의 조건수 | $\dfrac{\lVert\Delta\mathbf{x}\rVert}{\lVert\mathbf{x}\rVert} \le \kappa(\mathbf{A})\dfrac{\lVert\Delta\mathbf{b}\rVert}{\lVert\mathbf{b}\rVert}$, $\kappa(\mathbf{A}) = \lVert\mathbf{A}\rVert\,\lVert\mathbf{A}^{-1}\rVert$ | 2-노름이면 $\kappa = \sigma_{\max}/\sigma_{\min}$ | [9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md) |
| 오차와 안정성 | forward error $\lesssim \kappa \times$ backward error | backward stable 알고리즘이 기대할 수 있는 최선 | [9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md) |
| 전처리 | $\mathbf{M}^{-1}\mathbf{A}\mathbf{x} = \mathbf{M}^{-1}\mathbf{b}$ | $\mathbf{M} \approx \mathbf{A}$이고 역 적용이 쉬움 | [9.2](../../09-numerical-methods/02-conditioning-and-stability/index.md) |
| Newton 반복(스칼라) | $x_{k+1} = x_k - f(x_k)/f'(x_k)$, $e_{k+1} \approx \Big\lvert \dfrac{f''(x^*)}{2f'(x^*)} \Big\rvert e_k^2$ | 단순근 근처에서 이차 수렴 | [9.4](../../09-numerical-methods/04-nonlinear-equations/index.md) |
| Newton 반복(연립) | $\mathbf{J}(\mathbf{x}_k)\,\Delta\mathbf{x} = -\mathbf{F}(\mathbf{x}_k)$ | | [9.4](../../09-numerical-methods/04-nonlinear-equations/index.md) |
| 다항식 보간 오차 | $f(x) - p(x) = \dfrac{f^{(n+1)}(\xi)}{(n+1)!}\prod_{i=0}^n (x - x_i)$ | 등간격 고차 보간은 Runge 현상 | [9.5](../../09-numerical-methods/05-interpolation-and-quadrature/index.md) |
| 구적 규칙 | $\int_a^b f(x)\,dx \approx \sum_i w_i f(x_i)$ | 사다리꼴 $O(h^2)$, Simpson $O(h^4)$ | [9.5](../../09-numerical-methods/05-interpolation-and-quadrature/index.md) |
| 수렴 차수 측정 | $p \approx \log_2 \dfrac{E(h)}{E(h/2)}$ | 격자를 반으로 줄여 오차 비를 잰다 | [9.5](../../09-numerical-methods/05-interpolation-and-quadrature/index.md) |
| 차원의 저주 | 격자 규칙 오차 $O(N^{-p/d})$, Monte Carlo 오차 $O(N^{-1/2})$ | 점 $N$개, $d$차원 | [9.5](../../09-numerical-methods/05-interpolation-and-quadrature/index.md) |
| 유한차분 | 전진 $\dfrac{f(x + h) - f(x)}{h} = f'(x) + O(h)$, 중심 $\dfrac{f(x + h) - f(x - h)}{2h} = f'(x) + O(h^2)$ | $h$가 너무 작으면 반올림 오차가 커진다 | [8.4](../../08-calculus-and-autodiff/04-automatic-differentiation/index.md) |
| 2계 중심차분 | $\dfrac{u_{j+1} - 2u_j + u_{j-1}}{\Delta x^2} = \partial_{xx} u + O(\Delta x^2)$ | | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |
| 전진 Euler | $\mathbf{y}_{n+1} = \mathbf{y}_n + h\, f(t_n, \mathbf{y}_n)$ | 1차, 명시적 | [9.6](../../09-numerical-methods/06-ode-solvers/index.md) |
| 후진 Euler | $\mathbf{y}_{n+1} = \mathbf{y}_n + h\, f(t_{n+1}, \mathbf{y}_{n+1})$ | 1차, 암시적. stiff 문제에 쓴다 | [9.6](../../09-numerical-methods/06-ode-solvers/index.md) |
| 적응 step의 허용오차 | $\text{err}_i \le \text{atol}_i + \text{rtol}\cdot\lvert y_i \rvert$ | `solve_ivp`의 국소 오차 기준 | [9.6](../../09-numerical-methods/06-ode-solvers/index.md) |
| FTCS 안정 조건 | $r = \dfrac{\alpha\,\Delta t}{\Delta x^2} \le \dfrac{1}{2}$ | 열방정식 $\partial_t u = \alpha\,\partial_{xx} u$의 명시적 스킴. 9.7장은 확산계수를 $\alpha$로 쓴다 | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |
| CFL 조건 | $\nu = \dfrac{c\,\Delta t}{\Delta x} \le 1$ | 속도 $c$의 이류 방정식, 명시적 스킴 | [9.7](../../09-numerical-methods/07-pde-and-dae/index.md) |

**고전적 4단 Runge–Kutta(RK4)**([9.6](../../09-numerical-methods/06-ode-solvers/index.md)). 4차 방법이다.

$$
\begin{aligned}
\mathbf{k}_1 &= f(t_n, \mathbf{y}_n), & \mathbf{k}_2 &= f\big(t_n + \tfrac{h}{2}, \mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_1\big), \\
\mathbf{k}_3 &= f\big(t_n + \tfrac{h}{2}, \mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_2\big), & \mathbf{k}_4 &= f\big(t_n + h, \mathbf{y}_n + h\mathbf{k}_3\big), \\
\mathbf{y}_{n+1} &= \mathbf{y}_n + \tfrac{h}{6}\big(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\big). & &
\end{aligned}
$$

## 실습

이 부록의 공식은 손으로 외우는 대상이 아니라 **검증해서 쓰는** 대상이다. 다음 작업으로 공식을 자기 코드에서 믿을 수 있게 만든다.

1. **행렬 미분 항등식을 수치로 확인한다.** '미분과 행렬 미분' 표에서 세 개를 골라, 무작위 행렬에서 중심차분 gradient와 공식의 값을 비교한다. 상대 오차가 float64에서 $10^{-6}$ 아래인지 본다. 대칭이 아닌 $\mathbf{A}$로 이차형식을 시험해 $(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$와 $2\mathbf{A}\mathbf{x}$의 차이를 직접 확인한다.
2. **가우시안 조건부 공식을 표본으로 확인한다.** 3차원 가우시안에서 표본을 많이 뽑고, $x_3$가 특정 값 근처인 표본만 골라 $(x_1, x_2)$의 표본 평균·공분산을 조건부 공식과 비교한다.
3. **분포의 parameterization을 맞춘다.** 지수분포 rate $\lambda = 2$와 gamma($\alpha = 3$, rate $\beta = 2$)를 `scipy.stats`로 만들고 `.mean()`, `.var()`가 표의 값과 같은지 확인한다. 같지 않으면 rate와 scale을 바꿔 넣은 것이다.
4. **가우시안 KL의 닫힌 형태를 Monte Carlo로 확인한다.** $\mathrm{KL}(p \,\Vert\, q) = \mathbb{E}_{x \sim p}[\log p(x) - \log q(x)]$를 표본 평균으로 추정해 닫힌 형태와 비교하고, $p$와 $q$를 바꿔 비대칭을 본다.

## 실패 사례

- **다른 layout의 공식을 그대로 옮기는 것.** gradient를 행벡터로 쓰는 교재(denominator·numerator 규약이 섞인 자료)의 공식을 옮기면 전치가 하나씩 어긋난다. 옮긴 공식은 shape부터 맞춰 보고, 유한차분으로 한 번 검증한다([8.2](../../08-calculus-and-autodiff/02-matrix-calculus/index.md)).
- **이차형식 gradient를 대칭이라고 가정하는 것.** $\nabla_{\mathbf{x}}\,\mathbf{x}^\top\mathbf{A}\mathbf{x} = 2\mathbf{A}\mathbf{x}$는 $\mathbf{A}$가 대칭일 때만 맞다. 비대칭 행렬이 들어오는 코드에서는 조용히 틀린 gradient가 된다.
- **rate와 scale을 바꿔 넣는 것.** 지수분포·gamma의 rate $\beta$를 SciPy의 `scale`에 넣으면 평균이 $\alpha/\beta$ 대신 $\alpha\beta$가 된다. 표준편차와 분산(`norm(mu, sigma)`는 분산이 아니라 표준편차를 받는다)도 같은 종류의 실수다.
- **KL의 방향을 바꿔 쓰는 것.** $\mathrm{KL}(p \,\Vert\, q)$와 $\mathrm{KL}(q \,\Vert\, p)$는 값도 최적해의 성질도 다르다. MLE는 forward KL, 변분추론은 reverse KL을 줄인다([14.1](../../14-information-and-learning-theory/01-entropy-and-kl/index.md)).
- **nat와 bit를 섞는 것.** 엔트로피·상호정보량을 비교할 때 한쪽은 `np.log`, 다른 쪽은 `np.log2`로 계산하면 $\log 2 \approx 0.693$배 차이가 난다.
- **역행렬과 행렬식을 그대로 계산하는 것.** 정규방정식의 $(\mathbf{A}^\top\mathbf{A})^{-1}$은 조건수를 제곱하고, 고차원 $\det\boldsymbol{\Sigma}$는 overflow·underflow한다. 공식은 수학적 관계이고, 계산은 `solve`, Cholesky, `slogdet`로 한다([9.3](../../09-numerical-methods/03-numerical-linear-algebra/index.md)).
- **가정을 떼고 공식을 쓰는 것.** Hoeffding은 유계와 독립, $t$ 구간은 정규 모집단, Cramér–Rao는 불편추정량과 정칙 조건이 필요하다. 표의 '조건' 열을 함께 옮긴다.

## 통과 기준

- 표에서 고른 행렬 미분 항등식 세 개를 유한차분으로 검증하는 짧은 코드를 작성했고, 상대 오차가 $10^{-6}$ 아래다.
- `scipy.stats`로 만든 지수분포·gamma의 평균과 분산이 분포 표의 값과 일치한다.
- 두 일변수 가우시안의 KL을 닫힌 형태와 Monte Carlo로 계산해 Monte Carlo 표준오차 범위 안에서 일치함을 보였다.
