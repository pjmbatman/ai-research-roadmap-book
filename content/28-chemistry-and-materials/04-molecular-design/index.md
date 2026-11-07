---
id: chemistry-and-materials/molecular-design
title: "분자·재료 생성과 역설계"
title_en: "Generative Molecular and Materials Design"
tier: advanced
status: draft
prerequisites: [chemistry-and-materials/property-prediction, generative-and-foundation-models/evaluating-generative-models]
---

# 분자·재료 생성과 역설계

지금까지는 주어진 분자의 물성을 예측했다. **역설계**(inverse design)는 질문을 뒤집는다. 원하는 물성을 먼저 정하고, 그것을 가진 분자나 재료를 **찾아**낸다. 후보 공간이 천문학적으로 크기 때문에(약물 같은 유기 분자만 $10^{60}$개로 추정된다) 하나씩 뒤질 수 없고, 생성모델로 후보를 만들거나 최적화로 공간을 탐색한다.

이 장은 생성모델의 구조(19부: VAE, 자기회귀, 확산, flow)와 그 평가(19.7장)를 다시 가르치지 않는다. 여기서 가르치는 판단은 역설계에만 있는 함정이다. 핵심은 하나다. **물성 예측 모델을 목적함수로 최적화하면, 최적화기는 모델이 정확한 분자가 아니라 모델이 틀린(과신하는) 분자를 찾아낸다.** 생성 결과가 좋아 보이는 것과 실제로 좋은 것을 가르는 법, 그리고 합성 가능성과 다목적을 어떻게 평가에 넣는지를 합성 데이터로 확인한다.

## 학습 목표

- 생성모델로 후보를 설계한다
- 생성 결과를 화학적 타당성과 유용성으로 평가한다

## 생성 방식

**왜 필요한가.** 후보를 만드는 방식은 무엇을 쉽게·어렵게 만드는지를 정한다. 표현(28.1장)의 선택이 그대로 생성모델의 선택으로 이어진다.

**세 가지 표현, 세 가지 생성.**

- **문자열 생성**: SMILES를 토큰 열로 보고 자기회귀 모델(19.2장)이나 VAE(19.3장)로 생성한다(Gómez-Bombarelli 외, 2018). 구현이 단순하지만 문법적으로 틀린 문자열(닫히지 않은 고리, 맞지 않는 원자가)을 만들 수 있어 유효성이 낮을 수 있다.
- **그래프 생성**: 원자와 결합을 하나씩 더해 분자 그래프를 짓는다. 원자가 규칙을 생성 과정에 넣어 유효성을 높일 수 있지만, 생성 순서와 순열 대칭을 다뤄야 한다.
- **3차원 생성**: 원자 좌표를 직접 생성한다. 확산 모델(19.5장)과 등변 구조(24.8장)가 쓰인다. 결합 구조뿐 아니라 입체 배치가 중요한 물성(결합 포즈, 결정 구조)에 필요하다.

**조건부 생성.** 원하는 물성값을 조건으로 주어 그 물성을 가진 분자를 생성하게 할 수 있다(조건부 VAE, classifier guidance). 조건을 "강하게" 걸수록 생성 분포가 학습 분포에서 멀어져 유효성과 타당성이 떨어지는 긴장이 있다.

**최적화라는 대안.** 생성모델 없이, 알려진 분자에서 출발해 조금씩 바꿔 가며 목적함수를 최적화하는 방법도 널리 쓰인다. 유전 알고리즘(genetic algorithm)과 베이지안 최적화(Bayesian optimization)가 대표적이다. 이 장의 실습은 이 쪽을 쓴다. 합성 데이터로 최적화의 함정을 또렷이 보이기 위해서다.

## 합성 분자 공간

실제 분자 대신, 조각(fragment) 24종의 유무를 비트 벡터로 적은 장난감 분자 공간을 만든다. **참 물성**은 조각의 기여에서 조각이 몰릴 때의 혼잡(crowding) penalty를 뺀 값이고, **유효성**은 금지된 조각 쌍이 없고 조각 수가 1–8개인지로 정한다. **합성 가능성 점수**(SA score)는 조각이 많고 몰릴수록 커진다(합성이 어렵다). 참 규칙을 우리가 알기 때문에, 뒤에서 "모델이 틀린 것"과 "평가가 속은 것"을 구분할 수 있다.

```python {exec="browser" session="design"}
import numpy as np

L = 24                                             # 조각(fragment) 종류 수
rng = np.random.default_rng(0)
w = np.abs(rng.normal(0, 1.0, L)) + 0.5            # 조각마다 양의 물성 기여
crowd = np.abs(rng.normal(0, 0.6, (L, L))); crowd = np.triu(crowd, 1)   # 조각 쌍의 혼잡 penalty(양수)
incompat = {(2, 5), (7, 11), (13, 19)}             # 함께 있으면 안 되는 조각 쌍(유효성 규칙)

def valid(x):                                       # 화학적 타당성: 금지 쌍이 없고 조각 1–8개
    on = np.flatnonzero(x)
    if not (1 <= len(on) <= 8):
        return False
    return not any((i, j) in incompat for i in on for j in on if i < j)

def true_property(x):                               # 참 물성: 조각 기여 − 혼잡 penalty(조각이 몰리면 감소)
    return x @ w - x @ crowd @ x

def sa_score(x):                                    # 합성 가능성: 조각이 많을수록, 쌍이 몰릴수록 어렵다
    return 1.0 + 0.5 * x.sum() + 0.4 * (x @ (crowd > 0).astype(float) @ x)   # 1(쉬움)~10(어려움)

rng2 = np.random.default_rng(1)
sample = (rng2.random((2000, L)) < 0.15).astype(float)
val = np.array([valid(x) for x in sample])
print(f"무작위 비트 벡터 2000개 중 유효한 분자: {val.sum()} ({val.mean():.0%})")
props = np.array([true_property(x) for x in sample[val]])
sas = np.array([sa_score(x) for x in sample[val]])
print(f"유효 분자의 참 물성 범위 [{props.min():.2f}, {props.max():.2f}], SA 점수 범위 [{sas.min():.2f}, {sas.max():.2f}]")
```

```output
무작위 비트 벡터 2000개 중 유효한 분자: 1819 (91%)
유효 분자의 참 물성 범위 [-10.70, 5.55], SA 점수 범위 [1.50, 16.20]
```

## 최적화와 탐색

**유전 알고리즘.** 유전 알고리즘은 후보 집단을 평가해 상위를 부모로 남기고, 교차(crossover)와 돌연변이(mutation)로 자식을 만들어 다음 세대를 이룬다. 목적함수만 정하면 미분이 필요 없어 이산적인 분자 공간에 바로 쓸 수 있다. 목적함수에 **합성 가능성**을 넣느냐로 결과가 어떻게 달라지는지 본다.

```python {exec="browser" session="design"}
def ga(seed, obj, pop=60, gens=40):
    r = np.random.default_rng(seed)
    P = (r.random((pop, L)) < 0.12).astype(float)
    for _ in range(gens):
        f = np.array([obj(x) for x in P])
        P = P[np.argsort(f)[::-1]]
        parents = P[:pop // 2]                        # 상위 절반이 부모
        kids = []
        for _ in range(pop - len(parents)):
            a, b = parents[r.integers(len(parents))], parents[r.integers(len(parents))]
            child = np.where(r.random(L) < 0.5, a, b).copy()   # 균일 교차
            flip = r.random(L) < 0.06                 # 돌연변이
            child[flip] = 1 - child[flip]
            kids.append(child)
        P = np.vstack([parents, kids])
    return P[0]

prop_only = lambda x: true_property(x) if valid(x) else -1e9
with_sa = lambda x: true_property(x) - 0.8 * sa_score(x) if valid(x) else -1e9
for name, obj in [("물성만 최적화     ", prop_only), ("물성 − 0.8×SA 최적화", with_sa)]:
    winners = [ga(s, obj) for s in range(5)]
    p = np.mean([true_property(x) for x in winners])
    sa = np.mean([sa_score(x) for x in winners])
    print(f"{name}: 평균 물성 {p:+.2f}, 평균 SA 점수 {sa:.2f}")
```

```output
물성만 최적화     : 평균 물성 +6.87, 평균 SA 점수 8.00
물성 − 0.8×SA 최적화: 평균 물성 +5.95, 평균 SA 점수 3.70
```

물성만 최적화하면 평균 물성 +6.87의 분자를 찾지만 SA 점수가 8.00으로 합성이 어렵다. SA penalty를 넣으면 물성이 +5.95로 조금 낮아지는 대신 SA가 3.70으로 떨어져 훨씬 만들기 쉬운 분자가 된다. **역설계의 목적함수는 거의 언제나 다목적**이다. 물성 하나만 극대화한 후보는 합성 불가능하거나 독성이 있거나 불안정할 수 있다. 어떤 축을 목적함수에 넣고 어떤 가중치를 줄지가 설계 문제의 핵심이다.

**베이지안 최적화.** 실제 측정(실험·고비용 계산)이 비싸면, 측정 횟수를 아끼는 것이 중요하다. 베이지안 최적화는 대리 모델(흔히 Gaussian process, 16장)로 아직 측정하지 않은 후보의 물성과 **불확실성**을 예측하고, 획득 함수(acquisition function)로 다음에 측정할 후보를 고른다. 획득 함수는 "예측값이 높은 후보"(착취, exploitation)와 "불확실성이 큰 후보"(탐색, exploration)를 함께 고려한다. 25.3장의 실험 설계·능동 학습과 같은 원리다. 예측값만 보고 고르면(순수 착취) 다음 절의 함정에 빠진다.

::quiz{id="multiobjective"}

## 평가

**유효성·고유성·신규성.** 생성모델의 기본 지표는 세 가지다(19.7장). **유효성**(validity)은 생성물이 화학 규칙을 지키는 비율, **고유성**(uniqueness)은 생성물 중 서로 다른 것의 비율(같은 분자만 반복하면 낮다), **신규성**(novelty)은 학습 데이터에 없던 것의 비율이다. 셋을 함께 봐야 한다. 유효성만 높이려 학습 분자를 그대로 복사하면 신규성이 0이고, 신규성만 높이면 유효성이 무너진다.

**예측 모델 착취.** 역설계의 가장 중요한 함정이다. 물성 예측 모델을 목적함수로 최적화하면, 최적화기는 **모델이 높게 예측하는** 후보를 찾는다. 그런데 모델이 높게 예측하는 곳은 "실제로 물성이 높은 곳"일 수도 있고 "모델이 틀려서 높게 예측하는 곳"일 수도 있다. 최적화기는 둘을 구분하지 않으므로, 모델의 적용 영역(28.2장) 밖에서 모델이 과신하는 후보로 끌려간다.

조각 3개 이하의 "단순한" 분자만으로 학습한 선형 대리 모델(혼잡 penalty를 거의 못 본다)을 유전 알고리즘으로 최적화해 본다.

```python {exec="browser" session="design"}
from numpy.linalg import lstsq

n_frag = sample[val].sum(1)
lib = sample[val][n_frag <= 3][:400]               # 라이브러리: 조각 3개 이하의 '단순한' 분자만
y_lib = np.array([true_property(x) for x in lib])
w_hat, *_ = lstsq(np.c_[lib, np.ones(len(lib))], y_lib, rcond=None)   # 선형 대리 모델(혼잡은 거의 못 봄)
surrogate = lambda X: np.atleast_2d(X) @ w_hat[:L] + w_hat[L]

def tanimoto_max(X, ref):                           # ref와의 최대 Tanimoto 유사도
    inter = np.atleast_2d(X) @ ref.T
    return (inter / (np.atleast_2d(X).sum(1)[:, None] + ref.sum(1)[None, :] - inter + 1e-9)).max(1)

sur_obj = lambda x: surrogate(x)[0] if valid(x) else -1e9
opt = ga(0, sur_obj)
print(f"최적화된 분자의 조각 수: {int(opt.sum())} (라이브러리는 3개 이하)")
print(f"대리 모델이 예측한 물성: {surrogate(opt)[0]:+.2f}")
print(f"그 분자의 참 물성     : {true_property(opt):+.2f}"
      f"  (예측이 참보다 {surrogate(opt)[0] - true_property(opt):+.2f} 낙관)")
print(f"학습 라이브러리와의 최대 Tanimoto 유사도: {tanimoto_max(opt, lib)[0]:.2f}")
```

```output
최적화된 분자의 조각 수: 8 (라이브러리는 3개 이하)
대리 모델이 예측한 물성: +9.23
그 분자의 참 물성     : +2.32  (예측이 참보다 +6.91 낙관)
학습 라이브러리와의 최대 Tanimoto 유사도: 0.37
```

최적화기는 조각 8개를 가진 분자(라이브러리는 3개 이하)로 갔다. 대리 모델은 혼잡 penalty를 본 적이 없어 이 분자를 +9.23으로 예측하지만 참값은 +2.32다. **예측이 참보다 +6.91이나 낙관적**이고, 그 분자는 라이브러리와의 유사도가 0.37로 적용 영역 밖이다. 최적화기가 모델의 외삽 오류를 "발견"해 착취한 것이다. 이것이 생성 분자의 예측 물성을 그대로 믿으면 안 되는 이유다(Renz 외, 2019).

::quiz{id="predictor-exploitation"}

**적용 영역 필터.** 착취를 막는 한 방법은, 후보를 예측값으로 고르기 전에 적용 영역 밖 후보를 걸러 내는 것이다. 후보 풀에서 예측 상위를 그냥 고를 때와, 라이브러리 유사도가 높은(적용 영역 안) 후보만 추린 뒤 고를 때를 비교한다.

```python {exec="browser" session="design"}
pool = sample[val][n_frag >= 2][:1000]              # 측정 안 한 후보 풀(조각 수 다양)
pred = surrogate(pool)
true = np.array([true_property(x) for x in pool])
sim = tanimoto_max(pool, lib)                       # 각 후보의 라이브러리 유사도

top = np.argsort(pred)[::-1][:20]                   # 예측 상위 20개를 '유망 후보'로 선별
print(f"예측 상위 20개 후보: 평균 예측 {pred[top].mean():+.2f}, 평균 참 물성 {true[top].mean():+.2f}"
      f"  → 예측이 {pred[top].mean() - true[top].mean():+.2f} 낙관(착취)")
thr = np.quantile(sim, 0.7)                         # 적용 영역: 유사도 상위 30%
idx = np.flatnonzero(sim >= thr)
top_dom = idx[np.argsort(pred[idx])[::-1][:20]]
print(f"적용 영역 안에서 상위 20개:  평균 예측 {pred[top_dom].mean():+.2f}, 평균 참 물성 {true[top_dom].mean():+.2f}"
      f"  → 낙관 편향 {pred[top_dom].mean() - true[top_dom].mean():+.2f}")
print(f"예측 상위 20개 중 적용 영역 밖 비율: {(sim[top] < thr).mean():.0%}")
```

```output
예측 상위 20개 후보: 평균 예측 +6.02, 평균 참 물성 +3.09  → 예측이 +2.92 낙관(착취)
적용 영역 안에서 상위 20개:  평균 예측 +4.13, 평균 참 물성 +4.10  → 낙관 편향 +0.03
예측 상위 20개 중 적용 영역 밖 비율: 95%
```

예측값만으로 고른 상위 20개는 평균적으로 참값보다 +2.92 낙관적이고, 그중 95%가 적용 영역 밖이다. 적용 영역 안에서만 고르면 예측값은 낮아지지만(+6.02 → +4.13) 참값은 오히려 높고(+3.09 → +4.10), 낙관 편향이 +0.03으로 거의 사라진다. **예측값이 가장 높은 후보가 실제로 가장 좋은 후보가 아니다.** 통계적 결정에서 "추정값이 가장 큰 항목을 고르면 그 추정값은 대개 과대평가"라는 optimizer's curse(승자의 저주)의 화학판이고, 적용 영역 제약이 그 편향을 줄인다.

:::note[분야 예시]
신약 개발에서 생성모델로 수만 개의 후보를 만들고 활성 예측 모델로 점수를 매겨 상위를 합성·시험하면, 상위 후보가 예측만큼 활성을 보이지 않는 일이 흔하다. 그래서 실무에서는 예측값과 함께 합성 가능성, 유사 화합물의 실험 이력, 모델 불확실성을 함께 보고 후보를 거른다. 재료 탐색에서도 형성 에너지 예측으로 안정한 결정을 찾을 때, 예측 상위 후보를 그대로 믿지 않고 DFT로 재계산해 확인하는 단계를 둔다. 생성·예측은 후보를 좁히는 도구이지 최종 판정이 아니다.
:::

## 실험 검증

역설계의 출력은 "합성해 볼 가치가 있는 후보 목록"이지 "발견"이 아니다. 후보는 반드시 더 믿을 수 있는 방법(고수준 계산, 합성·측정)으로 검증해야 하고, 그 결과를 다시 모델에 넣는 **닫힌 고리**(closed-loop)로 돌릴 때 비로소 생성모델이 탐색을 이끈다. 이때 각 단계의 성공률을 기록하는 것이 중요하다. "생성한 후보 중 합성 가능한 비율", "합성한 것 중 목표 물성을 보인 비율"이 모델의 실제 가치를 말한다. 벤치마크의 예측 점수가 높은 것과 이 성공률이 높은 것은 다르다(28.5장).

## 실습

합성 분자 공간에서 생성·최적화·평가의 전 과정을 다룬다. 본문 코드가 유전 알고리즘, 다목적, 예측 모델 착취, 적용 영역 필터를 보인다. 실습에서는 다음을 바꿔 본다.

- **SA penalty 가중치**를 0에서 키우며 물성과 합성 가능성의 **Pareto front**(한쪽을 좋게 하려면 다른 쪽을 포기해야 하는 경계)를 그린다.
- **대리 모델의 학습 분포**를 넓히거나(조각 수 제한을 풀기) 불확실성 penalty를 목적함수에 더해, 착취가 얼마나 줄어드는지 측정한다.
- **생성 후보의 신규성**(라이브러리와의 최대 유사도)과 **참 물성**을 함께 그려, 신규성이 높을수록 예측이 믿을 수 없어지는 경향을 확인한다.

연습문제로 예측 모델 착취를 수치로 측정하는 함수를 구현한다.

::exercise{id="optimizer-curse"}

## 실패 사례

- **물성 예측 모델의 예측값을 생성 후보의 물성으로 보고하는 것.** 최적화기는 모델이 과신하는 적용 영역 밖 후보로 간다. 예측값은 후보를 거르는 신호일 뿐, 검증 결과가 아니다.
- **적용 영역을 보지 않고 예측 상위 후보를 선별하는 것.** 예측 상위는 대개 적용 영역 밖에 몰려 있다(optimizer's curse). 유사도나 불확실성으로 먼저 거른다.
- **물성 하나만 목적함수로 쓰는 것.** 합성 가능성, 안정성, 독성을 빼면, 최적 후보가 만들 수 없거나 쓸 수 없는 분자가 된다.
- **유효성·고유성·신규성 중 하나만 보고하는 것.** 셋은 맞바꿈 관계라, 하나만 높이면 다른 것이 무너진다.
- **검증 없이 생성 결과를 발견으로 보고하는 것.** 후보는 더 믿을 수 있는 방법으로 확인해야 하고, 성공률(합성 가능 비율, 목표 물성 달성 비율)로 평가한다.

## 통과 기준

- 생성 후보를 유효성·고유성·신규성, 합성 가능성, 적용 영역(유사도·불확실성), 검증 결과로 보고한다.
- 예측 모델을 목적함수로 쓸 때, 선별한 후보의 예측값과 (검증 가능하다면) 참값의 차이로 착취 여부를 점검한다.
- 연습문제 `optimizer-curse`의 테스트를 통과한다.

## 참고 문헌

- Gómez-Bombarelli 외 (2018). Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules. *ACS Central Science*, 4(2), 268–276. — SMILES VAE의 연속 잠재공간에서의 분자 생성·최적화
- Brown, Fiscato, Segler, Vaucher (2019). GuacaMol: Benchmarking Models for de Novo Molecular Design. *Journal of Chemical Information and Modeling*, 59(3), 1096–1108. — 생성 모델 벤치마크(유효성·신규성·다목적 최적화)
- Ertl, Schuffenhauer (2009). Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. *Journal of Cheminformatics*, 1, 8. — 합성 가능성 점수(SA score)
- Renz, Van Rompaey, Wegner, Hochreiter, Klambauer (2019). On failure modes in molecule generation and optimization. *Drug Discovery Today: Technologies*, 32–33, 55–63. — 생성·최적화에서 예측 모델 착취 등 실패 양상
