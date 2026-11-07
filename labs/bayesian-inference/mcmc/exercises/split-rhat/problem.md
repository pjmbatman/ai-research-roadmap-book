# split R-hat 구현하기

체인 $M$개를 각각 앞뒤 절반으로 나누면 길이 $n$인 체인 $2M$개가 된다. 나눈 체인 $j$의 평균을 $\bar\theta_j$, 표본분산(분모 $n-1$)을 $s_j^2$라 하면

$$
W = \frac{1}{2M}\sum_j s_j^2, \qquad
B = \frac{n}{2M - 1}\sum_j \big(\bar\theta_j - \bar\theta\big)^2, \qquad
\widehat{R} = \sqrt{\frac{\frac{n-1}{n}W + \frac{1}{n}B}{W}}
$$

이다. $\bar\theta$는 나눈 체인 평균들의 평균이다.

`split_rhat(chains)`를 완성한다.

- `chains`는 shape `(M, N)`인 배열(또는 중첩 리스트)이다. `np.asarray`로 바꾼다.
- 2차원이 아니거나 $N < 4$이면 `ValueError`를 낸다.
- $N$이 홀수이면 마지막 표본을 버리고 반으로 나눈다.

테스트는 세 가지를 확인한다. 같은 분포에서 독립으로 뽑은 체인은 $\widehat{R} \approx 1$이다. 서로 다른 곳에 머문 체인은 $\widehat{R}$가 크다. **체인이 하나뿐이어도** 앞뒤 절반이 다르면(추세가 있으면) split 덕분에 $\widehat{R}$가 커진다.
