---
id: statistical-inference/comparing-models
title: "모델 비교의 통계"
title_en: "Statistics for Comparing Models"
tier: core
status: draft
prerequisites: [statistical-inference/intervals-and-testing]
---

# 모델 비교의 통계

ML 논문의 결과 표는 대개 방법마다 숫자 하나를 적고 가장 좋은 숫자를 굵게 표시한다. 그런데 학습은 무작위 초기화, 데이터 순서, 데이터 분할, dropout 같은 무작위성을 거치므로, 같은 코드를 다시 돌리면 다른 숫자가 나온다. 표의 두 숫자 차이가 이 무작위성보다 작다면, 굵은 글씨는 방법의 차이가 아니라 운을 표시한 것일 수 있다.

이 장은 12.4장의 구간과 검정을 모델 비교에 적용한다. 변동이 어디서 오는지 나누어 재고, 같은 조건에서 짝지은 비교로 작은 차이를 드러내고, 교차검증과 여러 데이터셋에서의 비교가 빠지기 쉬운 함정을 피하고, 마지막으로 결과를 독자가 판단할 수 있게 보고하는 형식을 정한다.

## 학습 목표

- seed·데이터 분할에 따른 변동을 보고한다
- 짝지은 비교와 적절한 검정을 사용한다

## 변동의 출처

모델의 성능 숫자 하나는 여러 무작위 선택의 결과다. 어떤 선택이 바뀌면 숫자가 얼마나 바뀌는지에 따라 출처를 나눈다.

| 출처 | 무엇이 바뀌는가 | 통제·측정 방법 |
|---|---|---|
| **학습 seed** | 가중치 초기화, mini-batch 순서, dropout mask, 데이터 증강 | seed를 바꿔 여러 번 학습 |
| **데이터 분할** | 어떤 샘플(또는 셀·환자)이 학습·시험에 들어가는가 | 분할을 바꿔 반복, 교차검증 |
| **하이퍼파라미터 탐색** | 탐색의 무작위성, 탐색 예산 | 방법마다 같은 예산, 탐색까지 포함해 반복 |
| **하드웨어·소프트웨어 비결정성** | GPU의 병렬 덧셈 순서, 라이브러리의 알고리즘 선택 | 결정적 설정(성능 비용 있음), 환경 기록(22.2장) |
| **데이터셋 자체** | 같은 분포에서 다른 표본을 모았다면 | 새 데이터 수집, 여러 데이터셋 비교 |

하드웨어 비결정성은 부동소수점 덧셈의 결합법칙이 성립하지 않는 데서 온다(9.1장). GPU에서 병렬로 더하는 순서가 실행마다 달라지면 결과의 마지막 자릿수가 달라지고, 이 작은 차이가 학습을 거치며 커질 수 있다. seed를 고정해도 결과가 완전히 같지 않은 이유다.

이 출처들이 실제로 얼마나 큰지 확인한다. 배터리 셀 30개에서 셀마다 40번 측정한 회귀 데이터를 만든다. 셀마다 모델이 설명할 수 없는 편차가 있어서, 어떤 셀이 시험 세트에 들어가는지가 점수에 영향을 준다. 방법은 무작위 특징(random features) ridge 회귀 두 가지다. A는 특징 30개, B는 특징 100개를 쓰고, 학습 seed가 무작위 특징을 정한다. 분할은 셀 단위로 30%를 시험에 둔다.

```python {exec="browser" session="cmp"}
import numpy as np

def make_data(rng, n_cells=30, per_cell=40, freq=2.0):   # 셀 30개 × 측정 40개
    x = rng.uniform(-2, 2, size=(n_cells * per_cell, 2))                  # (N, 2)
    cell = np.repeat(np.arange(n_cells), per_cell)                         # (N,)
    offset = rng.normal(0, 0.3, n_cells)[cell]                             # 셀마다 다른 편차
    y = np.sin(freq * x[:, 0]) * np.cos(x[:, 1]) + offset + rng.normal(0, 0.2, len(x))
    return x, y, cell

def fit_predict(x_tr, y_tr, x_te, seed, n_feat, scale=1.5):   # 무작위 특징 ridge 회귀
    r = np.random.default_rng(seed)                          # 학습 seed가 특징을 정한다
    W, b = r.normal(0, scale, (2, n_feat)), r.uniform(0, 2 * np.pi, n_feat)
    P = np.cos(x_tr @ W + b)                                 # (N_tr, F)
    w = np.linalg.solve(P.T @ P + 1e-3 * len(y_tr) * np.eye(n_feat), P.T @ y_tr)
    return np.cos(x_te @ W + b) @ w

def group_split(cell, split_seed, test_frac=0.3):          # 셀 단위 분할
    cells = np.random.default_rng(split_seed).permutation(np.unique(cell))
    test = np.isin(cell, cells[: int(len(cells) * test_frac)])
    return ~test, test

x, y, cell = make_data(np.random.default_rng(0))
def rmse(n_feat, seed, split):
    tr, te = group_split(cell, split)
    return np.sqrt(np.mean((fit_predict(x[tr], y[tr], x[te], seed, n_feat) - y[te]) ** 2))

print(f"데이터 {len(y)}개, 셀 {cell.max() + 1}개")
print(f"seed 0, 분할 0:  A(특징 30개) RMSE {rmse(30, 0, 0):.4f}   B(특징 100개) RMSE {rmse(100, 0, 0):.4f}")
```

```output
데이터 1200개, 셀 30개
seed 0, 분할 0:  A(특징 30개) RMSE 0.4265   B(특징 100개) RMSE 0.3849
```

실행 한 번으로는 B가 0.042만큼 낫다. 이제 seed 10개와 분할 10개의 모든 조합(100번)에서 두 방법을 학습하고, seed만 바꿀 때와 분할만 바꿀 때의 표준편차를 따로 잰다.

```python {exec="browser" session="cmp"}
S = 10
R = {f: np.array([[rmse(f, s, sp) for s in range(S)] for sp in range(S)]) for f in (30, 100)}   # (분할, seed)
for f, name in ((30, "A"), (100, "B")):
    r = R[f]
    print(f"{name}: 평균 {r.mean():.4f}   seed만 바꿀 때 sd {r.std(axis=1, ddof=1).mean():.4f}"
          f"   분할만 바꿀 때 sd {r.std(axis=0, ddof=1).mean():.4f}")

D = R[100] - R[30]                                        # 같은 (분할, seed)에서의 차이 B - A
print(f"\n짝지은 차이 B - A: 평균 {D.mean():+.4f}, sd {D.std(ddof=1):.4f}   B가 이긴 조합 {np.mean(D < 0):.0%}")
print("분할 0에서 seed별 차이:", " ".join(f"{v:+.4f}" for v in D[0]))
```

```output
A: 평균 0.4110   seed만 바꿀 때 sd 0.0109   분할만 바꿀 때 sd 0.0666
B: 평균 0.4021   seed만 바꿀 때 sd 0.0013   분할만 바꿀 때 sd 0.0678

짝지은 차이 B - A: 평균 -0.0089, sd 0.0107   B가 이긴 조합 78%
분할 0에서 seed별 차이: -0.0416 -0.0054 -0.0241 +0.0012 -0.0084 -0.0091 -0.0000 -0.0182 -0.0125 +0.0004
```

세 가지가 보인다.

- **분할 변동이 seed 변동보다 훨씬 크다.** 어떤 셀이 시험에 들어가느냐에 따른 표준편차(약 0.067)가 seed에 따른 표준편차(A 0.011, B 0.001)의 6배 이상이다. seed만 여러 개 돌리고 분할을 하나로 고정하면 가장 큰 변동 출처를 보지 못한다.
- **방법마다 seed 민감도가 다르다.** 특징이 적은 A는 어떤 특징이 뽑히느냐에 크게 좌우되고, B는 거의 영향을 받지 않는다. 안정성 자체가 방법의 성질이므로 함께 보고할 가치가 있다.
- **처음 실행의 차이(0.042)는 평균 차이(0.009)의 네 배가 넘었다.** 100개 조합 중 22%에서는 A가 이겼다. 분할 0 하나에서도 seed에 따라 차이가 $-0.042$부터 $+0.001$까지 흩어진다. 단일 실행의 차이로 우열을 주장하면 크기와 방향 모두 틀릴 수 있다.

::quiz{id="single-seed-claim"}

:::note[분야 예시]
신약 후보 분자의 성질 예측 벤치마크에서는 분자를 무작위로 나누는 분할과, 분자 골격(scaffold)이 같은 분자를 한쪽에 몰아넣는 분할이 흔히 함께 쓰인다. 골격 단위 분할은 "처음 보는 골격"으로의 일반화를 묻는 것이고, 어떤 골격이 시험에 들어가느냐에 따라 점수가 크게 흔들린다. 이 장의 셀 단위 분할과 같은 구조다. 분할 방식과 분할 seed를 함께 보고하지 않은 점수는 비교하기 어렵다.
:::

## 짝지은 비교

앞의 결과에서 분할 변동(0.067)은 두 방법의 차이(0.009)보다 일곱 배 이상 크다. 그런데도 100개 조합 중 78%에서 B가 이겼다. 분할이 바뀔 때 두 방법의 점수가 **함께** 오르내리기 때문이다. 어려운 셀이 시험에 들어가면 A와 B가 함께 나빠진다. 이 공통 변동은 같은 조건에서의 차이를 계산하면 상쇄된다. 이것이 **짝지은 비교**(paired comparison)다.

$$
\operatorname{Var}(e_B - e_A) = \operatorname{Var}(e_A) + \operatorname{Var}(e_B) - 2\operatorname{Cov}(e_A, e_B).
$$

두 모델의 오차가 양의 상관을 가지면 공분산 항이 차이의 분산을 줄인다. 짝지을 수 있는 단위는 여러 층에 있다.

- **같은 시험 샘플**: 샘플마다 두 모델의 오차 차이를 계산한다.
- **같은 분할·같은 seed**: 실행마다 두 방법의 점수 차이를 계산한다(앞 셀의 `D`).
- **같은 데이터셋**: 여러 데이터셋에서 데이터셋마다 차이를 본다(이 장의 뒤 절).

시험 샘플 단위로 확인한다. 분할 0, seed 0의 시험 샘플마다 두 모델의 제곱오차를 구하고, MSE 차이의 표준오차를 세 가지로 계산한다.

```python {exec="browser" session="cmp"}
tr, te = group_split(cell, 0)
ea = (fit_predict(x[tr], y[tr], x[te], 0, 30) - y[te]) ** 2       # 샘플별 제곱오차, (N_te,)
eb = (fit_predict(x[tr], y[tr], x[te], 0, 100) - y[te]) ** 2
d, n = eb - ea, len(ea)
cd = np.array([d[cell[te] == c].mean() for c in np.unique(cell[te])])   # 셀별 평균 차이, (C_te,)

print(f"MSE 차이 B - A = {d.mean():+.5f},  두 모델 오차의 상관 {np.corrcoef(ea, eb)[0, 1]:.2f}")
print(f"짝 없는 SE         {np.sqrt(ea.var(ddof=1) / n + eb.var(ddof=1) / n):.5f}")
print(f"짝지은 SE (샘플)   {d.std(ddof=1) / np.sqrt(n):.5f}")
print(f"짝지은 SE (셀 {len(cd)}개) {cd.std(ddof=1) / np.sqrt(len(cd)):.5f}")
```

```output
MSE 차이 B - A = -0.03373,  두 모델 오차의 상관 0.76
짝 없는 SE         0.01317
짝지은 SE (샘플)   0.00672
짝지은 SE (셀 9개) 0.00666
```

두 모델의 샘플별 오차는 상관계수 0.76으로 함께 움직이고, 짝을 지으면 표준오차가 절반으로 준다. 같은 시험 세트에서 두 모델을 평가했다면 짝지은 비교를 쓰지 않을 이유가 없다.

::quiz{id="why-pairing"}

셀 단위 표준오차도 확인했다. 시험 샘플은 셀 9개에서 나왔으므로, 결론을 새 셀로 일반화하려면 12.4장처럼 셀이 독립 단위다. 이 예에서는 셀 단위와 샘플 단위의 표준오차가 거의 같다. 셀마다 다른 편차가 두 모델에 똑같이 더해져 차이에서 상쇄되었기 때문이다. 12.4장의 실습처럼 **모델의 차이 자체**가 셀마다 다르면 셀 단위 표준오차가 훨씬 커진다. 어느 경우인지는 미리 알 수 없으므로 독립 단위로 계산해 본다.

**paired bootstrap.** 공식 대신 부트스트랩을 쓸 수도 있다. 샘플(또는 셀)을 복원추출할 때 두 모델의 오차를 **같은 인덱스로** 함께 뽑으면 짝이 유지된다. 이 방식은 MSE 차이뿐 아니라 AUC 차이, F1 차이처럼 샘플별 값의 평균이 아닌 metric에도 그대로 쓸 수 있다. 분류 문제에서 두 분류기의 맞고 틀림을 짝지어 비교하는 McNemar 검정도 같은 생각이다(Dietterich, 1998).

## 교차검증 결과의 비교

교차검증(cross-validation, CV)은 데이터를 $k$개 fold로 나누어 각 fold를 한 번씩 시험에 쓰므로, 한 데이터셋에서 점수 $k$개를 얻는다. 여기서 흔한 실수가 나온다. 점수 $k$개(또는 반복 CV의 $r \times k$개)를 독립 표본으로 보고 표준오차 $\mathrm{sd}/\sqrt{rk}$나 $t$ 검정을 쓰는 것이다.

그러나 이 점수들은 독립이 아니다. 5-fold에서 두 fold의 학습 데이터는 75%가 겹치고, 모든 점수가 같은 데이터셋 하나에서 나왔다. Bengio와 Grandvalet(2004)은 $k$-fold CV 추정값의 분산에 대해 모든 분포에서 성립하는 불편 추정량이 없음을 보였고, 학습 데이터의 겹침에서 오는 오차 사이의 상관을 무시하는 순진한 추정량은 분산을 크게 과소평가할 수 있다고 지적했다. 반복 횟수를 늘리면 문제가 더 분명해진다. 같은 분포에서 뽑은 독립 데이터셋 100개 각각에서 반복 group 5-fold CV를 하고, "순진한" 표준오차와 CV 추정값이 데이터셋에 따라 실제로 흔들리는 정도를 비교한다.

```python {exec="browser" session="cmp"}
def cv_scores(x, y, cell, reps, k=5):                     # 반복 group k-fold, 점수 reps·k개
    out = []
    for r in range(reps):
        fold = np.random.default_rng(r).permutation(np.unique(cell)) % k
        for j in range(k):
            te = fold[cell] == j
            out.append(np.sqrt(np.mean((fit_predict(x[~te], y[~te], x[te], r, 30) - y[te]) ** 2)))
    return np.array(out)

for reps in (1, 4, 16):
    est, naive = [], []
    for ds in range(100):                                 # 같은 분포에서 뽑은 독립 데이터셋 100개
        s = cv_scores(*make_data(np.random.default_rng(1000 + ds), n_cells=20, per_cell=10), reps)
        est.append(s.mean())
        naive.append(s.std(ddof=1) / np.sqrt(len(s)))
    print(f"반복 {reps:2d}번 (점수 {5 * reps:2d}개):  순진한 SE 평균 {np.mean(naive):.4f}   "
          f"CV 추정값의 실제 sd {np.std(est, ddof=1):.4f}")
```

```output
반복  1번 (점수  5개):  순진한 SE 평균 0.0343   CV 추정값의 실제 sd 0.0461
반복  4번 (점수 20개):  순진한 SE 평균 0.0170   CV 추정값의 실제 sd 0.0436
반복 16번 (점수 80개):  순진한 SE 평균 0.0083   CV 추정값의 실제 sd 0.0426
```

반복을 1번에서 16번으로 늘리자 순진한 표준오차는 0.034에서 0.008로 4분의 1이 되었다. 그러나 CV 추정값이 데이터셋에 따라 흔들리는 실제 정도는 약 0.043으로 거의 그대로다. 반복은 분할 방식의 운을 평균할 뿐, **데이터셋이 하나라는 사실**에서 오는 불확실성은 줄이지 못한다. 반복 횟수로 나눈 표준오차는 연구자가 계산 예산을 늘리는 것만으로 얼마든지 작게 만들 수 있는 숫자다.

::quiz{id="repeated-cv-se"}

두 방법을 CV로 비교할 때도 같은 문제가 생긴다. Dietterich(1998)는 여러 검정의 1종 오류를 시뮬레이션으로 비교해, 무작위 분할을 반복한 뒤 차이에 $t$ 검정을 하는 방법은 1종 오류가 높아 쓰지 말아야 하고, 10-fold CV 차이의 $t$ 검정도 1종 오류가 다소 높다고 보고했다. 대안으로 5×2 CV 검정과 McNemar 검정을 제안했다. Nadeau와 Bengio(2003)는 학습 데이터의 겹침을 반영하도록 분산 추정을 보정한 검정을 제안했다. 실무에서는 다음을 지킨다.

- 반복 CV 점수의 개수로 표준오차를 나누지 않는다.
- 검정을 쓴다면 겹침을 고려한 방법(5×2 CV, 보정된 resampled $t$ 검정)을 쓰고, 어떤 방법인지 밝힌다.
- 가장 믿을 만한 근거는 탐색에 쓰지 않은 **독립 시험 데이터**에서의 짝지은 비교다. 가능하면 CV는 모델 선택에, 최종 비교는 별도 시험 세트에 쓴다(16.6장).

## 여러 데이터셋에서의 비교

새 방법을 여러 벤치마크 데이터셋에서 평가할 때, 데이터셋별 점수를 평균해 비교하는 것은 문제가 있다. 데이터셋마다 점수의 척도가 달라 점수가 큰 데이터셋 하나가 평균을 지배하고, 데이터셋 사이의 변동은 정규분포를 따른다고 보기 어렵다. Demšar(2006)는 두 방법의 비교에는 Wilcoxon signed-rank 검정을, 여러 방법의 비교에는 **순위 기반** 방법을 권했다.

1. 데이터셋마다 방법들에 순위를 매긴다(1 = 최고, 동점이면 평균 순위).
2. 방법마다 평균 순위 $R_j$를 구한다.
3. **Friedman 검정**으로 "모든 방법의 평균 순위가 같다"는 귀무가설을 검정한다. $N$개 데이터셋, $k$개 방법에서 통계량은
   $$
   \chi^2_F = \frac{12N}{k(k+1)}\left[\sum_{j=1}^k R_j^2 - \frac{k(k+1)^2}{4}\right]
   $$
   이고, 귀무가설에서 근사적으로 자유도 $k-1$인 $\chi^2$ 분포를 따른다.
4. 기각되면 **Nemenyi 사후 검정**으로 쌍마다 비교한다. 두 방법의 평균 순위 차이가 **critical difference**
   $$
   \mathrm{CD} = q_\alpha\sqrt{\frac{k(k+1)}{6N}}
   $$
   보다 크면 차이가 유의하다. $q_\alpha$는 studentized range 분포에서 나온 값으로, $\alpha = 0.05$에서 $k = 3$이면 2.343이다.

목표 함수의 진동수가 다른 데이터셋 12개에서 세 방법을 비교한다. C는 특징 100개지만 낮은 진동수 위주로 특징을 뽑는다(`scale=0.5`). 자유도가 2이면 $\chi^2$ 분포의 꼬리 확률이 $e^{-x/2}$이므로 p-value를 바로 계산할 수 있다.

```python {exec="browser" session="cmp"}
from math import exp, sqrt

methods = {"A 특징 30": (30, 1.5), "B 특징 100": (100, 1.5), "C 특징 100, 좁은 대역": (100, 0.5)}
freqs = np.linspace(0.5, 3.0, 12)                          # 데이터셋 12개: 목표 함수의 진동수
scores = np.zeros((len(freqs), len(methods)))              # RMSE, (N_ds, k)
for i, fq in enumerate(freqs):
    xd, yd, cd_ = make_data(np.random.default_rng(i), freq=fq)
    tr, te = group_split(cd_, 0)
    for j, (nf, sc) in enumerate(methods.values()):
        scores[i, j] = np.sqrt(np.mean((fit_predict(xd[tr], yd[tr], xd[te], 0, nf, sc) - yd[te]) ** 2))

ranks = scores.argsort(axis=1).argsort(axis=1) + 1         # 데이터셋마다 순위 (1 = 최고), 동점 없음
N, k = scores.shape
avg = ranks.mean(axis=0)
chi2 = 12 * N / (k * (k + 1)) * (np.sum(avg**2) - k * (k + 1) ** 2 / 4)   # Friedman 통계량
cd_nemenyi = 2.343 * sqrt(k * (k + 1) / (6 * N))            # q_0.05 = 2.343 (k = 3)
for name, a, m in zip(methods, avg, scores.mean(axis=0)):
    print(f"{name:20s} 평균 순위 {a:.2f}   평균 RMSE {m:.4f}")
print(f"Friedman χ² = {chi2:.2f}, p = {exp(-chi2 / 2):.4f} (자유도 2)   Nemenyi CD = {cd_nemenyi:.2f}")
```

```output
A 특징 30              평균 순위 1.83   평균 RMSE 0.4236
B 특징 100             평균 순위 1.58   평균 RMSE 0.3911
C 특징 100, 좁은 대역      평균 순위 2.58   평균 RMSE 0.4412
Friedman χ² = 6.50, p = 0.0388 (자유도 2)   Nemenyi CD = 0.96
```

Friedman 검정은 세 방법의 순위가 같다는 가설을 기각한다(p = 0.039). Nemenyi 비교에서 B와 C의 평균 순위 차이(1.00)는 CD(0.96)보다 커서 유의하지만, A와 B의 차이(0.25)와 A와 C의 차이(0.75)는 CD보다 작다. 평균 RMSE로는 B가 A보다 0.03 낮아 분명해 보이지만, 데이터셋 12개의 순위로는 A와 B를 구별할 수 없다. 데이터셋 수가 적으면 CD가 크다는 점에 주의한다. CD는 $1/\sqrt{N}$에 비례하므로 데이터셋을 4배로 늘려야 절반이 된다. 결과는 평균 순위를 축에 놓고 CD보다 가까운 방법들을 선으로 묶은 **critical difference 그림**으로 보여 주는 것이 관례다(Demšar, 2006).

::quiz{id="critical-difference"}

## 보고 방식

독자가 결과를 판단하려면 숫자 외에 그 숫자가 어떻게 만들어졌는지가 필요하다. Bouthillier 외(2021)는 데이터 샘플링·초기화·하이퍼파라미터 선택에서 오는 변동이 벤치마크 결과에 무시할 수 없는 영향을 주며, 여러 변동 출처를 함께 무작위화해 비교하는 편이 낫다고 보고했다. 결과 표에는 최소한 다음을 쓴다.

- **실행 횟수와 무엇을 바꿨는지**: "seed 5개 × 셀 단위 5-fold", "분할 고정, seed 10개" 등.
- **중심값과 퍼짐**: 평균과 표준편차. 표준편차인지 표준오차인지 표의 캡션에 밝힌다. 표준편차는 "한 번 돌리면 얼마나 흔들리는가", 표준오차($\mathrm{sd}/\sqrt{n}$)는 "평균이 얼마나 정확한가"다.
- **비교의 근거**: 짝지은 차이의 평균과 신뢰구간, 사용한 검정과 독립 단위.
- **선택 과정**: 하이퍼파라미터 탐색 예산이 방법 사이에 공정했는지, 몇 가지 설정을 시도했는지(12.4장의 선택 편향).

::quiz{id="sd-vs-se"}

::quiz{id="se-from-sd"}

예를 들어 다음 두 표는 같은 실험을 보고한다.

| 나쁜 예 | RMSE |
|---|---|
| A | 0.4265 |
| B | **0.3849** |

| 좋은 예 | RMSE (평균 ± 표준편차, seed 5개 × 셀 단위 5-fold) |
|---|---|
| A | 0.414 ± 0.015 |
| B | 0.403 ± 0.005 |
| B − A (짝지은 차이) | −0.012, 95% 구간 [−0.030, +0.007], 5/5 seed에서 B 우세 |

그림으로는 평균 막대 하나보다 **실행마다의 점**을 모두 찍고, 같은 seed끼리 선으로 이어 짝을 보여 주는 편이 정보가 많다. 점이 다섯 개뿐이라면 그 사실 자체가 보여야 한다.

:::note[분야 예시]
기상·기후 예측에서는 새 모델을 운영 모델과 비교할 때 같은 초기 시각들에서의 예측 오차를 짝지어 비교하는 것이 관행이다. 날씨에 따라 두 모델이 함께 잘 맞히거나 함께 틀리기 때문이다. 다만 연속된 날의 오차는 서로 상관되어 있어(날씨 패턴이 며칠씩 이어짐) 날짜 하나하나를 독립 표본으로 보면 유의성을 과대평가한다. 짝을 짓는 단위와 독립 단위를 따로 생각해야 하는 예다.
:::

## 실습

두 방법을 seed 5개와 셀 단위 5-fold 분할로 비교하고, B가 낫다고 주장할 수 있는지 판단한다. 데이터와 방법은 본문과 같고, seed마다 fold 배정과 무작위 특징이 함께 바뀐다. 코드는 `labs/statistical-inference/comparing-models/compare_methods.py`다. 세 질문에 차례로 답한다. 실행마다 얼마나 흔들리는가, 같은 조건에서의 짝지은 차이는 얼마인가, 그 차이가 새 셀에도 일반화되는가. 집필 시점의 실행 결과는 다음과 같다.

```output
seed   A RMSE   B RMSE   B - A
   0   0.4356   0.3993   -0.0363
   1   0.4081   0.4045   -0.0036
   2   0.4203   0.4056   -0.0146
   3   0.3965   0.3962   -0.0003
   4   0.4110   0.4083   -0.0027

1. 방법별 요약 (seed 5개)
   A: 평균 0.4143   표준편차 0.0146   표준오차 0.0065
   B: 평균 0.4028   표준편차 0.0049   표준오차 0.0022

2. 짝지은 차이 B - A (같은 seed, 같은 fold)
   평균 -0.0115   표준편차 0.0149   95% 구간 (t, 자유도 4) [-0.0300, +0.0070]   B가 이긴 seed 5/5
   짝을 짓지 않았다면 차이의 표준오차 0.0069 (짝지은 경우 0.0067)

3. 새 셀로의 일반화: 셀 30개를 재표본화한 짝지은 부트스트랩 (MSE 차이, seed 평균)
   MSE 차이 -0.00954   95% 구간 [-0.01243, -0.00670]   B가 나은 셀 27/30
```

결과를 읽는 법은 다음과 같다.

- **seed 수준에서는 결론이 약하다.** B는 5개 seed 모두에서 이겼지만, 짝지은 차이의 95% 구간 [−0.030, +0.007]은 0을 포함한다. seed 0에서 A가 유독 나빴기 때문이다. 5전 5승의 양측 부호 검정 p-value는 $2 \times (1/2)^5 = 0.0625$로, seed 5개로는 0.05 기준을 넘을 수 없다.
- **여기서는 짝을 지어도 seed 수준의 표준오차가 거의 줄지 않았다(0.0069 → 0.0067).** 앞 절의 고정 분할 실험과 달리, 교차검증은 seed마다 모든 셀을 한 번씩 시험에 쓰므로 seed 사이에 두 방법이 함께 흔들리는 공통 변동이 작다. 남은 변동은 주로 A의 seed 민감도이고, 이것은 짝을 지어도 상쇄되지 않는다. 짝지은 비교의 이득은 공통 변동이 있을 때만 생긴다.
- **데이터 단위에서는 근거가 강하다.** 셀마다 out-of-fold 오차를 seed 5개에 걸쳐 평균하고 셀을 재표본화하면, MSE 차이의 95% 구간이 0을 포함하지 않고 30개 셀 중 27개에서 B가 낫다. 다만 이 구간은 셀 표본의 변동을 반영할 뿐, seed 5개를 평균한 값 자체의 seed 변동은 반영하지 않는다.

판단: "B가 A보다 평균적으로 낫다"는 셀 단위 근거와 seed 방향의 일관성으로 지지된다. 그러나 "B가 RMSE를 0.012 줄인다"는 크기 주장은 seed 수준의 구간이 넓어 약하다. 크기를 주장하려면 seed를 늘리고 A의 seed 민감도를 함께 보고한다.

이어서 두 연습문제를 푼다. 첫째는 seed 결과를 표준편차·표준오차·짝지은 차이로 요약하고, 둘째는 여러 데이터셋에서의 평균 순위와 critical difference를 계산한다.

::exercise{id="seed-summary"}

::exercise{id="average-ranks"}

## 실패 사례

- **단일 seed 결과로 우열을 주장하는 것.** 이 장의 실험에서 첫 실행의 차이는 평균 차이의 네 배가 넘었고, 같은 두 방법에서 22%의 조합은 반대 결론을 냈다. seed·분할을 여러 개 쓰고 짝지은 차이를 보고한다.
- **표준편차와 표준오차를 혼동하는 것.** "0.411 ± 0.011"의 0.011이 어느 쪽인지 밝히지 않으면 독자는 정밀도를 $\sqrt{n}$배 잘못 읽는다. 캡션에 쓴다.
- **반복 CV 점수의 개수로 표준오차를 나누는 것.** 점수들은 같은 데이터에서 나와 상관되어 있다. 반복을 늘리면 이 숫자는 0으로 가지만 실제 불확실성은 줄지 않는다.
- **seed만 바꾸고 분할을 고정하는 것.** 이 장의 데이터에서는 분할 변동이 seed 변동보다 6배 이상 컸다. 가장 큰 변동 출처를 빠뜨리면 오차 막대가 실제보다 훨씬 좁다.
- **여러 데이터셋의 점수를 평균만 내는 것.** 척도가 큰 데이터셋이 평균을 지배한다. 순위 기반 비교와 데이터셋별 결과를 함께 보고한다.
- **방법마다 다른 탐색 예산을 쓰는 것.** 새 방법은 공들여 튜닝하고 baseline은 기본 설정으로 두면, 차이의 일부는 방법이 아니라 튜닝 노력이다.

## 통과 기준

- 자신의 실험 결과 표를 실행 횟수, 바꾼 무작위 요소, 평균 ± 표준편차(또는 표준오차, 어느 쪽인지 명시), 짝지은 차이와 그 구간까지 포함해 작성한다.
- 결과에 영향을 주는 변동 출처(seed·분할·탐색) 중 가장 큰 것을 실험으로 확인하고, 비교에 쓴 독립 단위를 밝힌다.
- 연습문제 두 개(`seed-summary`, `average-ranks`)의 테스트를 통과한다.

## 참고 문헌

- Dietterich (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923. — 분류 알고리즘 비교 검정들의 1종 오류 비교, 5×2 CV 검정
- Nadeau, Bengio (2003). Inference for the generalization error. *Machine Learning*, 52, 239–281. — 학습 데이터 겹침을 고려한 교차검증 분산 추정
- Bengio, Grandvalet (2004). No unbiased estimator of the variance of K-fold cross-validation. *Journal of Machine Learning Research*, 5, 1089–1105. — 교차검증 분산 추정의 한계
- Demšar (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30. — Wilcoxon·Friedman 검정, Nemenyi 사후 검정, critical difference 그림
- Bouthillier et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems 3* (MLSys 2021). — 벤치마크 결과에서 변동 출처의 영향과 비교 방법에 대한 권고
