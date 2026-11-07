---
id: life-sciences/protein-structure
title: "단백질 구조 예측과 설계"
title_en: "Protein Structure Prediction and Design"
tier: advanced
status: draft
prerequisites: [life-sciences/sequence-models, scientific-ml/geometric-models]
---

# 단백질 구조 예측과 설계

단백질의 기능은 3차원 구조에서 나온다. 실험으로 구조를 결정하는 일은 오래 걸리고 비싸며, 모든 단백질에 성공하지도 않는다. 그래서 서열에서 구조를 예측하는 문제는 수십 년 동안 계산생물학의 중심 문제였다. 2021년 AlphaFold는 많은 단백질에서 실험 구조에 가까운 정확도로 구조를 예측했고, 잔기마다 예측의 신뢰도를 함께 내놓았다(Jumper 외, 2021). 이제 예측 구조는 데이터베이스에 대량으로 쌓여 있고, 많은 연구가 예측 구조를 입력으로 쓴다.

이 장의 초점은 모델의 구조가 아니라 **판단**이다. 예측 구조와 실험 구조를 어떻게 비교하는가, 지표 하나가 무엇을 놓치는가, 신뢰도 점수를 어디까지 믿을 수 있는가, 그리고 예측이 답하지 못하는 질문은 무엇인가. 좌표를 다루는 데 필요한 대칭의 언어(회전·평행이동에 대한 불변성과 등변성)는 7.10장과 24.8장에서 다룬다.

## 학습 목표

- 구조 예측 문제와 평가 지표를 이해한다
- 예측 구조의 신뢰도를 해석한다
- 회전·평행이동에 불변인 구조 비교(거리 행렬, Kabsch 정렬 RMSD)를 구현한다

## 구조 예측 문제

**왜 필요한가.** 구조를 알면 결합 부위, 변이의 영향, 다른 단백질과의 접촉면을 추론할 수 있다. 구조 예측은 이 추론을 실험 구조가 없는 단백질로 넓힌다.

**정의.** 단백질 구조 예측은 아미노산 서열 $s = (s_1, \dots, s_L)$에서 원자 좌표를 예측하는 문제다. 이 장에서는 잔기마다 대표 원자 하나, 알파 탄소(Cα)의 좌표만 다룬다. 구조는 $\mathbf{X} \in \mathbb{R}^{L \times 3}$(단위 Å)이다. 좌표 자체에는 임의성이 있다. 구조 전체를 회전하거나 평행이동해도 같은 구조다. 그래서 출력은 "좌표"가 아니라 **회전·평행이동을 제외한 모양**이다.

**거리 행렬과 접촉 지도.** 회전·평행이동에 영향받지 않는 표현은 잔기 쌍 사이의 거리 행렬 $D_{ij} = \lVert \mathbf{x}_i - \mathbf{x}_j \rVert$다. 거리 행렬을 문턱값으로 자른 이진 행렬을 **접촉 지도**(contact map)라 한다. 흔히 쓰는 정의는 잔기의 Cβ 원자(글리신은 Cα) 사이 거리가 8 Å 미만인 쌍이다. 여기서는 Cα만 있으므로 Cα 거리로 대신한다. 서열상 가까운 잔기는 당연히 가까우므로, 정보가 있는 것은 서열상 멀리 떨어진(예: $|i - j| \ge 6$) 잔기 사이의 접촉이다. 29.2장의 공진화 신호가 예측하려던 것이 바로 이 장거리 접촉이다.

이상적인 α-나선 두 개를 고리로 이은 합성 구조로 확인한다. 나선의 Cα는 반지름 2.3 Å, 잔기당 1.5 Å 상승, 100° 회전의 나선 위에 놓인다.

```python {exec="browser" session="struct"}
import numpy as np

def helix(n, start, sign):                             # 이상적인 α-나선의 Cα 좌표 (Å)
    t = np.arange(n)
    ang = np.deg2rad(100.0) * t                        # 잔기당 100° 회전
    return np.stack([2.3 * np.cos(ang), 2.3 * np.sin(ang), 1.5 * t * sign], 1) + start

h1 = helix(20, [0.0, 0.0, 0.0], +1)                    # (20, 3)
h2 = helix(20, [10.0, 0.0, 28.5], -1)                  # (20, 3) 반대 방향의 두 번째 나선
loop = h1[-1] + np.linspace(0, 1, 5)[1:-1, None] * (h2[0] - h1[-1])   # (3, 3)
X = np.vstack([h1, loop, h2])                          # (43, 3) Å

def dist_matrix(X):
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)   # (L, L)

D = dist_matrix(X)
sep = np.abs(np.subtract.outer(np.arange(len(X)), np.arange(len(X))))   # (L, L) 서열 간격 |i-j|
contact = (D < 8.0) & (sep >= 6)                       # 서열상 멀리 떨어진 접촉
print("나선 안 이웃 Cα 거리:", np.round(np.linalg.norm(np.diff(h1, axis=0), axis=1)[:3], 2), "Å")
print("장거리 접촉 쌍 수:", contact.sum() // 2)

rng = np.random.default_rng(0)
Rq, _ = np.linalg.qr(rng.normal(size=(3, 3)))          # 무작위 직교 행렬
X_moved = X @ Rq.T + np.array([5.0, -3.0, 12.0])       # 회전 + 평행이동
print("이동 후 좌표 최대 변화:", round(float(np.abs(X_moved - X).max()), 1), "Å")
print("이동 후 거리 행렬 변화 < 1e-8:", np.abs(dist_matrix(X_moved) - D).max() < 1e-8)
```

```output
나선 안 이웃 Cα 거리: [3.83 3.83 3.83] Å
장거리 접촉 쌍 수: 22
이동 후 좌표 최대 변화: 35.2 Å
이동 후 거리 행렬 변화 < 1e-8: True
```

나선 안에서 이웃한 Cα 사이 거리는 3.83 Å로 일정하다. 두 나선이 나란히 놓여 장거리 접촉이 22쌍 생긴다. 구조를 회전·평행이동하면 좌표는 최대 35 Å 넘게 바뀌지만 거리 행렬은 수치 오차 범위에서 그대로다.

**진화 정보와 기하 제약.** 현대 구조 예측 모델의 입력은 대개 두 가지 정보원을 쓴다. 하나는 29.2장의 진화 정보다. 같은 가족 서열들의 다중 서열 정렬이나 그 정보를 흡수한 언어모델 표현이다. 다른 하나는 기하 제약이다. 거리는 삼각 부등식을 만족해야 하고, 이웃한 Cα 사이는 약 3.8 Å이며, 원자끼리 겹칠 수 없다. 출력이 회전·평행이동에 대해 올바르게 변환되도록 모델을 설계하는 것은 24.8장의 등변 모델이 다루는 주제다.

:::note[분야 예시]
**구조생물학**에서는 예측 구조를 X선 결정학의 위상 결정이나 cryo-EM 밀도 지도 해석의 출발 모델로 쓴다. 실험 데이터가 최종 판단을 하고 예측은 탐색을 줄인다. **진화생물학**에서는 서열 유사성이 거의 남지 않은 먼 친척 단백질도 구조는 닮은 경우가 많아, 예측 구조를 비교해 기능을 추론한다. **약물 개발**에서는 결합 부위의 모양을 보고 후보 분자를 거른다. 이때는 결합 부위 몇 Å의 오차가 결론을 바꿀 수 있어 전체 정확도보다 국소 정확도가 중요하다.
:::

## 평가 지표

**왜 필요한가.** 예측 구조가 실험 구조와 "얼마나 같은가"를 하나의 숫자로 요약해야 모델을 비교할 수 있다. 그런데 구조는 회전·평행이동의 자유가 있고, 오차가 한 곳에 몰리기도 하고 고르게 퍼지기도 한다. 지표마다 이 둘을 다르게 취급한다.

**RMSD와 Kabsch 정렬.** 예측 $\mathbf{P}$와 기준 $\mathbf{Q}$(둘 다 $L \times 3$, 같은 잔기끼리 대응)의 RMSD는 최적의 회전 $\mathbf{R}$과 평행이동 $\mathbf{t}$로 겹친 뒤의 평균 제곱 거리의 제곱근이다.

$$
\mathrm{RMSD}(\mathbf{P}, \mathbf{Q}) = \min_{\mathbf{R} \in SO(3),\, \mathbf{t}} \sqrt{\frac{1}{L} \sum_{i=1}^{L} \lVert \mathbf{R}^\top \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \rVert^2 }.
$$

평행이동의 최적값은 두 무게중심을 맞추는 것이다. 회전은 Kabsch 알고리즘이 닫힌 형태로 준다(Kabsch, 1976). 중심을 뺀 좌표로 $3 \times 3$ 행렬 $\mathbf{H} = \mathbf{P}_0^\top \mathbf{Q}_0$을 만들고 SVD $\mathbf{H} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$(7.6장)를 구하면, 행벡터 좌표에 오른쪽에서 곱하는 최적 회전은

$$
\mathbf{R} = \mathbf{U} \,\mathrm{diag}(1, 1, d)\, \mathbf{V}^\top, \qquad d = \operatorname{sign}\det(\mathbf{U}\mathbf{V}^\top)
$$

이다. $d$는 결과가 반사(행렬식 $-1$)가 되지 않게 막는다. 반사를 허용하면 거울상 구조도 RMSD 0으로 겹쳐진다. 단백질은 거울상이 다른 분자이므로 반사는 허용하지 않는다.

```python {exec="browser" session="struct"}
def kabsch(P, Q, allow_reflection=False):              # P @ R + t가 Q에 가장 가깝도록
    P0, Q0 = P - P.mean(0), Q - Q.mean(0)              # (L, 3) 중심을 원점으로
    U, S, Vt = np.linalg.svd(P0.T @ Q0)                # 3x3 공분산의 SVD
    d = 1.0 if allow_reflection else np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt                # (3, 3)
    rmsd = np.sqrt(np.mean(np.sum((P0 @ R - Q0) ** 2, axis=1)))
    return rmsd, R

def plain_rmsd(P, Q):
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))

model = X_moved + rng.normal(0, 0.5, X.shape)          # 좌표마다 0.5 Å 잡음
rmsd, R = kabsch(model, X)
print(f"정렬 없는 RMSD   {plain_rmsd(model, X):6.2f} Å")
print(f"Kabsch 정렬 RMSD {rmsd:6.2f} Å,  det(R) = {np.linalg.det(R):.3f}")

mirror = X * np.array([1.0, 1.0, -1.0])                # 거울상: 거리 행렬은 같다
print("거울상 거리 행렬 변화 < 1e-8:", np.abs(dist_matrix(mirror) - D).max() < 1e-8)
print(f"거울상, 반사 허용  RMSD {kabsch(mirror, X, allow_reflection=True)[0]:5.2f} Å")
print(f"거울상, 회전만     RMSD {kabsch(mirror, X)[0]:5.2f} Å")
```

```output
정렬 없는 RMSD    22.49 Å
Kabsch 정렬 RMSD   0.80 Å,  det(R) = 1.000
거울상 거리 행렬 변화 < 1e-8: True
거울상, 반사 허용  RMSD  0.00 Å
거울상, 회전만     RMSD  3.17 Å
```

정렬하지 않은 RMSD 22.49 Å은 구조의 차이가 아니라 회전·평행이동의 차이다. Kabsch 정렬 후의 0.80 Å은 좌표마다 넣은 0.5 Å 잡음의 크기(원자당 $0.5\sqrt{3} \approx 0.87$ Å)와 비슷하다. 거울상 구조는 거리 행렬이 완전히 같다. 그래서 거리 행렬만 보는 지표는 거울상을 구분하지 못한다. Kabsch 정렬에서 반사를 허용하면 RMSD가 0이 되고, 회전만 허용해야 3.17 Å의 차이가 드러난다.

::quiz{id="mirror-image"}

**RMSD의 약점.** RMSD는 제곱 평균이라 큰 오차 몇 개에 지배된다. 두 도메인 각각은 정확한데 도메인 사이의 각도만 틀리면, 한쪽에 맞춘 중첩에서 다른 쪽이 통째로 어긋나 RMSD가 커진다. 또 RMSD의 크기는 단백질 길이에 따라 달라 서로 다른 단백질 사이에서 비교하기 어렵다.

**TM-score.** TM-score는 잔기별 거리를 길이에 따라 정한 척도 $d_0$로 눌러 0과 1 사이로 만든다(Zhang·Skolnick, 2004).

$$
\mathrm{TM} = \max_{\mathbf{R}, \mathbf{t}} \frac{1}{L} \sum_{i=1}^{L} \frac{1}{1 + (d_i / d_0)^2}, \qquad d_0(L) = 1.24 \sqrt[3]{L - 15} - 1.8.
$$

$d_i$는 중첩 후 $i$번째 잔기의 거리, $L$은 기준 구조의 길이다. 먼 잔기는 기여가 0에 가까워질 뿐 점수를 무한히 깎지 않으므로, 맞는 부분이 많을수록 점수가 높다. 최댓값은 RMSD를 최소화하는 중첩이 아니라 **TM-score를 최대화하는 중첩**에서 얻는다. 아래 코드는 길이 8의 구간마다 Kabsch 중첩을 해 보고 최댓값을 취하는 단순한 근사라, 실제 TM-score의 하한이다.

::quiz{id="tm-score-d0"}

**lDDT.** lDDT는 중첩을 하지 않는다(Mariani 외, 2013). 기준 구조에서 일정 반경(기본값 15 Å) 안에 있는 원자 쌍들의 거리가 예측 구조에서 얼마나 보존되는지를, 네 가지 허용 오차(0.5, 1, 2, 4 Å) 안에 드는 비율의 평균으로 잰다. 거리만 비교하므로 회전·평행이동에 불변이고, 도메인이 움직여도 각 도메인 안의 국소 구조가 맞으면 점수가 높다. 원래 정의는 모든 원자를 쓰고 같은 잔기 안의 쌍을 빼며 잔기별 점수도 낸다. 아래 코드는 Cα만 쓴 단순화판이다.

두 가지 오차를 비교한다. 두 번째 나선을 고리 시작점을 축으로 35° 돌린 "경첩 운동" 모델과, 모든 잔기에 1.3 Å 잡음을 고르게 넣은 모델이다.

```python {exec="browser" session="struct"}
def tm_score(P, Q):                                    # 여러 부분 구간에서 시작한 중첩 중 최댓값(하한)
    L = len(Q)
    d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8
    best = 0.0
    for s in range(0, L - 7):                          # 길이 8 구간마다 Kabsch 중첩
        _, R = kabsch(P[s:s + 8], Q[s:s + 8])
        Pa = (P - P[s:s + 8].mean(0)) @ R + Q[s:s + 8].mean(0)
        best = max(best, np.mean(1 / (1 + (np.linalg.norm(Pa - Q, axis=1) / d0) ** 2)))
    return best

def lddt_ca(P, Q, r0=15.0):                            # Cα만 쓴 단순화 lDDT (중첩 없음)
    Dp, Dq = dist_matrix(P), dist_matrix(Q)
    pair = (Dq < r0) & ~np.eye(len(Q), dtype=bool)     # 기준 구조에서 15 Å 안의 쌍
    diff = np.abs(Dp - Dq)[pair]
    return np.mean([np.mean(diff < t) for t in (0.5, 1.0, 2.0, 4.0)])

theta = np.deg2rad(35.0)                               # 고리 시작점을 축으로 두 번째 나선을 35° 회전
Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
hinge = X.copy()
hinge[20:] = (X[20:] - X[20]) @ Ry.T + X[20]
noisy = X + rng.normal(0, 1.3, X.shape)                # 전체에 고르게 퍼진 오차
for name, P in [("경첩 운동", hinge), ("전체 잡음", noisy)]:
    print(f"{name}: RMSD {kabsch(P, X)[0]:4.2f} Å  TM-score {tm_score(P, X):.2f}  lDDT(Cα) {lddt_ca(P, X):.2f}")
```

```output
경첩 운동: RMSD 5.10 Å  TM-score 0.60  lDDT(Cα) 0.67
전체 잡음: RMSD 2.36 Å  TM-score 0.47  lDDT(Cα) 0.57
```

RMSD로는 경첩 운동 모델(5.10 Å)이 전체 잡음 모델(2.36 Å)보다 두 배 넘게 나쁘다. TM-score와 lDDT로는 순서가 뒤집힌다. 경첩 운동 모델은 두 나선 각각의 모양이 완벽하고 상대 배치만 틀렸다. 전체 잡음 모델은 모든 곳이 조금씩 틀렸다. 어느 쪽이 "더 좋은 예측"인지는 질문에 달려 있다. 결합 부위가 한 도메인 안에 있다면 경첩 모델이 더 유용하고, 도메인 사이의 배치가 기능의 핵심이라면 어느 쪽도 충분하지 않다. **지표 하나로 순위를 매기지 말고, 전역 지표와 국소 지표를 함께 보고한다.**

::widget{name="kabsch-align"}

::quiz{id="domain-hinge"}

**신뢰도 점수의 해석.** 구조 예측 모델은 예측과 함께 신뢰도를 낸다. AlphaFold의 pLDDT는 잔기마다 0–100 척도로, 그 잔기 주변의 예측이 실험 구조와 lDDT 기준으로 얼마나 맞을지를 추정한 값이다(Jumper 외, 2021). 신뢰도 점수는 **모델 자신의 추정**이다. 다른 모델 출력과 마찬가지로 실제 오차와 얼마나 맞는지(보정, 21.2장)를 확인해야 쓸 수 있다. 확인하는 방법은 실험 구조가 있는 단백질에서 잔기별 신뢰도와 잔기별 실제 오차를 비교하는 것이다. 합성 예측으로 절차를 보인다. 고리 주변과 끝부분의 오차를 크게 두었고, 신뢰도는 실제 오차 크기에 잡음을 섞어 만들었다.

```python {exec="browser" session="struct"}
from scipy.stats import spearmanr

sigma = np.full(len(X), 0.3)                           # (43,) 잔기별 실제 오차 크기 (Å)
sigma[18:26] = 2.5                                     # 고리 주변은 불확실
sigma[-5:] = 4.0                                       # 끝부분은 거의 모름
pred = X + rng.normal(size=X.shape) * sigma[:, None]
conf = np.clip(100 - 22 * sigma + rng.normal(0, 5, len(X)), 0, 100)   # 모델이 낸 신뢰도 점수

def lddt_per_res(P, Q, r0=15.0):                       # 잔기별 단순화 lDDT
    Dp, Dq = dist_matrix(P), dist_matrix(Q)
    pair = (Dq < r0) & ~np.eye(len(Q), dtype=bool)
    ok = np.mean([(np.abs(Dp - Dq) < t) for t in (0.5, 1.0, 2.0, 4.0)], axis=0)   # (L, L)
    return np.array([ok[i, pair[i]].mean() for i in range(len(Q))])

rho = spearmanr(conf, lddt_per_res(pred, X)).statistic
hi = conf >= 70
print(f"신뢰도와 잔기별 lDDT의 Spearman 상관: {rho:.2f}")
print(f"전체 {len(X)}개 잔기 RMSD {kabsch(pred, X)[0]:.2f} Å")
print(f"신뢰도 70 이상 {hi.sum()}개 잔기 RMSD {kabsch(pred[hi], X[hi])[0]:.2f} Å")
```

```output
신뢰도와 잔기별 lDDT의 Spearman 상관: 0.64
전체 43개 잔기 RMSD 2.23 Å
신뢰도 70 이상 30개 잔기 RMSD 0.52 Å
```

신뢰도와 잔기별 lDDT의 순위 상관은 0.64다. 신뢰도가 낮은 잔기를 빼면 RMSD가 2.23 Å에서 0.52 Å으로 줄어든다. 두 가지를 읽을 수 있다. 첫째, 신뢰도는 **어느 부분을 믿을지** 정하는 데 쓴다. 신뢰도가 높은 부분의 구조는 해석할 수 있고, 낮은 부분은 구조로 해석하지 않는다. 둘째, 신뢰도로 걸러 낸 뒤의 RMSD를 전체 성능처럼 보고하면 안 된다. 걸러 낸 잔기 수와 함께 보고한다. 실제 단백질에서 신뢰도가 낮은 영역은 모델이 모르는 영역일 수도 있고, 실제로 일정한 구조가 없는 유연한 영역일 수도 있다. 두 경우는 신뢰도만으로 구분되지 않는다.

::quiz{id="low-confidence-region"}

## 설계

**왜 필요한가.** 구조 예측은 서열 → 구조 방향이다. 단백질 설계는 반대 방향이다. 원하는 구조나 기능을 가진 서열을 찾는다.

**구조 기반 서열 설계.** 주어진 뼈대(backbone) 구조에 접힐 서열을 찾는 문제를 역접힘(inverse folding)이라 한다. 학습된 모델은 구조를 입력받아 위치마다 아미노산 분포를 내고, 여기서 서열을 뽑는다. 같은 구조에 접히는 서열은 많으므로 이 문제는 하나의 답이 아니라 분포를 다룬다.

**생성모델 기반 설계.** 확산 모델(19.5장) 같은 생성모델로 새로운 뼈대 구조나 서열·구조 쌍을 직접 생성하는 방법도 쓰인다. 결합 부위나 대칭 같은 조건을 걸어 생성할 수 있다.

**설계의 평가.** 설계에는 정답 구조가 없다. 그래서 흔히 두 단계를 거친다.

1. **계산 검증(in silico).** 설계한 서열을 구조 예측 모델에 넣어, 예측 구조가 목표 구조와 맞는지(RMSD, TM-score)와 신뢰도를 본다. 이를 자기 일관성(self-consistency) 검사라 부르기도 한다.
2. **실험 검증.** 실제로 단백질을 만들어 발현되는지, 접히는지, 원하는 기능이 있는지 측정한다.

계산 검증은 필요조건일 뿐이다. 설계 모델과 검증 모델이 같은 데이터로 학습되었거나 같은 편향을 공유하면, 계산 검증을 통과하는 비율은 실험 성공률보다 크게 높을 수 있다. 실험 검증으로 이어지는 평가 설계는 29.5장에서 다룬다.

## 한계

**동역학.** 단백질은 움직인다. 예측 모델은 대개 하나의 정적인 구조를 낸다. 기능이 구조 변화(열림·닫힘, 결합에 따른 변형)에 있다면 하나의 구조는 그 일부만 보여 준다.

**복합체와 리간드.** 많은 단백질은 다른 단백질, 핵산, 작은 분자, 금속 이온과 결합해야 기능한다. 단일 사슬 예측은 결합 상대가 있을 때의 모양을 보장하지 않는다. 복합체와 리간드를 함께 예측하는 모델도 있지만, 학습 데이터가 단일 사슬보다 적고 평가도 더 어렵다.

**변이의 효과.** 한 잔기 치환이 안정성이나 기능을 크게 바꾸어도 예측 구조는 거의 그대로일 수 있다. 구조 예측 모델의 학습 목표는 변이 효과의 크기를 맞히는 것이 아니다.

**예측과 실험의 간극.** 예측 구조 데이터베이스의 구조는 측정값이 아니다. 실험 구조도 결정 조건(결정 격자, 온도, 해상도)의 영향을 받은 하나의 관측이다. 예측 구조를 실험 구조처럼 학습 데이터에 섞으면, 다음 모델은 이전 모델의 오차까지 학습한다. 출처를 항상 구분해 기록한다.

::quiz{id="predicted-vs-experimental"}

## 실습

구조 예측 결과 몇 개를 실험 구조와 비교한다. 구조 파일을 읽고 잔기를 대응시키는 단계에는 Biopython이나 Gemmi 같은 구조 파일 라이브러리를 쓸 수 있다(집필 시점 예시. 이 책의 실습 환경에는 들어 있지 않으므로 각 라이브러리의 설치 안내를 확인한다).

1. 실험 구조가 있는 단백질 3–5개를 고르고, 데이터베이스 판과 식별자를 기록한다(29.1장).
2. 각 단백질의 예측 구조를 구한다(예측 구조 데이터베이스에서 내려받거나 직접 예측). 예측에 쓴 모델과 판을 기록한다.
3. 두 구조에서 같은 잔기의 Cα 좌표만 뽑아 $L \times 3$ 배열로 맞춘다. 실험 구조에서 빠진 잔기는 양쪽에서 뺀다.
4. Kabsch 정렬 RMSD, TM-score(공식 프로그램이 있으면 그것으로), Cα lDDT를 계산한다.
5. 잔기별 신뢰도와 잔기별 lDDT(또는 정렬 후 거리)의 순위 상관을 계산하고, 신뢰도 구간별 평균 오차를 표로 만든다.
6. 신뢰도가 낮은 영역이 실험 구조에서도 무질서한 영역(좌표가 없는 잔기)과 겹치는지 확인한다.

핵심 계산 두 가지를 직접 구현한다.

::exercise{id="kabsch-rmsd"}

::exercise{id="contact-map"}

## 실패 사례

- **예측 구조를 실험 구조처럼 사용하는 것.** 예측 구조로 결합 부위를 해석하거나 다른 모델의 학습 데이터로 쓸 때, 출처와 신뢰도를 기록하지 않으면 모델 오차가 사실처럼 전파된다.
- **신뢰도가 낮은 영역을 해석하는 것.** 신뢰도가 낮은 고리나 끝부분의 모양으로 기능을 추론하지 않는다. 그 영역은 모델이 모르거나 실제로 유연한 부분이다.
- **정렬하지 않고 좌표를 비교하는 것.** 좌표의 차이는 대부분 회전·평행이동의 차이다. 반대로 반사를 허용한 정렬은 거울상을 같은 구조로 판정한다.
- **RMSD 하나로 모델을 비교하는 것.** 도메인 배치 오차 하나가 RMSD를 지배한다. TM-score와 lDDT 같은 지표를 함께 본다.
- **신뢰도로 거른 잔기의 성능만 보고하는 것.** 거른 비율을 함께 쓰지 않으면 성능이 부풀려진다.

## 통과 기준

- 예측 구조와 실험 구조 3개 이상에 대해 Kabsch 정렬 RMSD, TM-score, lDDT를 표로 보고하고, 지표 사이에 순위가 다른 경우를 구조의 어느 부분 때문인지로 설명한다.
- 잔기별 신뢰도와 잔기별 오차의 순위 상관, 신뢰도 구간별 평균 오차를 보고하고, 해석에서 제외한 영역과 그 비율을 적는다.
- 연습문제 `kabsch-rmsd`와 `contact-map`의 테스트를 통과한다.

## 참고 문헌

- Kabsch (1976). A solution for the best rotation to relate two sets of vectors. *Acta Crystallographica Section A*, 32(5), 922–923. — 두 좌표 집합을 겹치는 최적 회전의 닫힌 형태 해
- Zhang, Skolnick (2004). Scoring function for automated assessment of protein structure template quality. *Proteins*, 57(4), 702–710. — 길이에 따른 척도 $d_0$를 쓰는 TM-score
- Mariani, Biasini, Barbato, Schwede (2013). lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests. *Bioinformatics*, 29(21), 2722–2728. — 중첩 없이 국소 거리 보존을 재는 lDDT
- Jumper 외 (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589. — AlphaFold와 잔기별 신뢰도 pLDDT
