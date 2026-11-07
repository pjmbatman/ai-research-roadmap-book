# 조각 단위로 읽으며 평균·분산 구하기

메모리에 다 올릴 수 없는 측정 데이터를 조각(chunk) 단위로 읽으며 전체 평균과 분산을 구하는 함수 `streaming_mean_var(chunks)`를 완성한다. `chunks`는 1차원 NumPy 배열을 하나씩 내보내는 iterable(보통 generator)이다.

```python
chunks = (np.load(f) for f in files)        # 한 번에 파일 하나만 메모리에
mean, var = streaming_mean_var(chunks)
```

요구 사항은 다음과 같다.

- 반환값 `(mean, var)`는 모든 조각을 이어 붙인 배열 `x`의 `x.mean()`, `x.var()`(모집단 분산, `ddof=0`)와 같아야 한다.
- 입력을 **한 번만** 순회한다. generator는 두 번 순회할 수 없다(5.1장). 조각을 리스트에 모아 두었다가 이어 붙이는 방법은 메모리 조건을 어긴다.
- 빈 조각이 섞여 있을 수 있다. 전체가 비어 있으면 `ValueError`를 낸다.
- **수치적으로 안정해야 한다.** 평균이 $10^8$처럼 크고 분산이 1 정도인 데이터에서도 정확해야 한다. 교과서 공식 $\mathrm{E}[x^2] - \mathrm{E}[x]^2$를 그대로 누적하면 두 큰 수의 차이에서 유효 숫자가 사라져 분산이 엉뚱한 값(심지어 음수)이 된다.

안정한 방법은 조각마다 `(개수, 평균, 편차 제곱합)`을 구한 뒤 다음 공식으로 합치는 것이다. 두 묶음 $a, b$에 대해

$$
n = n_a + n_b, \quad \delta = \bar{x}_b - \bar{x}_a, \quad \bar{x} = \bar{x}_a + \delta \frac{n_b}{n}, \quad M_2 = M_{2,a} + M_{2,b} + \delta^2 \frac{n_a n_b}{n}
$$

이고, 분산은 $M_2 / n$이다. 여기서 $M_2 = \sum_i (x_i - \bar{x})^2$는 편차 제곱합이다.
