# 접촉 지도와 접촉 예측의 정밀도

Cα 좌표 $\mathbf{X} \in \mathbb{R}^{L \times 3}$ (Å)에서 접촉 지도를 만들고, 예측한 접촉 점수가 실제 접촉을 얼마나 잘 맞히는지 잰다. 두 함수를 완성한다.

**1. `contact_map(X, cutoff=8.0, min_sep=6)`**

$L \times L$ bool 배열 $C$를 반환한다. $C_{ij}$는 다음 두 조건을 모두 만족할 때만 `True`다.

- 거리 $\lVert \mathbf{x}_i - \mathbf{x}_j \rVert$ **<** `cutoff`
- 서열 간격 $|i - j|$ **≥** `min_sep` (서열상 가까운 잔기는 당연히 가까우므로 뺀다)

$C$는 대칭이고 대각선은 `False`다.

**2. `top_l_precision(scores, true_contacts, min_sep=6)`**

`scores`는 예측 모델이 낸 $L \times L$ 접촉 점수(클수록 접촉일 가능성이 높음, 대칭), `true_contacts`는 `contact_map`의 결과다. 서열 간격이 `min_sep` 이상인 쌍 $(i < j)$ 가운데 점수가 높은 순서로 **상위 $L$개**를 골라, 그중 실제 접촉인 비율을 반환한다. 접촉 예측을 평가할 때 흔히 쓰는 "top-L 정밀도"다. 점수가 같은 쌍이 경계에 걸리는 경우는 테스트하지 않는다.

예: 모든 실제 접촉 쌍에 다른 쌍보다 높은 점수를 주고 실제 접촉 쌍이 $L$개 이상이면 정밀도는 1이다.

힌트: `np.triu_indices(L, k=min_sep)`는 $j - i \ge$ `min_sep`인 쌍의 인덱스를 준다.
