# STLSQ로 동역학 방정식 찾기

시간미분 $\dot{\mathbf{X}} \in \mathbb{R}^{T \times d}$와 후보 함수 라이브러리 $\mathbf{\Theta} \in \mathbb{R}^{T \times p}$가 주어졌을 때

$$
\dot{\mathbf{X}} \approx \mathbf{\Theta}\,\mathbf{\Xi}
$$

를 만족하는 **희소한** 계수 행렬 $\mathbf{\Xi} \in \mathbb{R}^{p \times d}$를 순차 임계 최소제곱(STLSQ)으로 구한다.

1. 최소제곱 해 $\mathbf{\Xi}$로 시작한다.
2. 절댓값이 `thresh`보다 작은 계수를 0으로 만든다.
3. 출력 열 $k$마다, 0이 아닌 계수에 해당하는 라이브러리 열만으로 최소제곱을 다시 풀어 그 계수들을 갱신한다.
4. 2–3을 `iters`번 반복한다.

함수 `stlsq(Theta, dX, thresh, iters=10)`를 완성해 `Xi`(shape `(p, d)`)를 반환한다. 한 번 0이 된 계수는 다시 살아나지 않는다(3단계에서 남은 열만 쓰므로).

테스트는 감쇠 진동자 $\dot{x} = -0.1x + 2y$, $\dot{y} = -2x - 0.1y$의 궤적과 2차까지의 단항식 라이브러리로, 참 계수 네 개만 남는지 확인한다.

힌트: `np.linalg.lstsq(A, b, rcond=None)[0]`. 열 선택에는 불리언 마스크를 쓴다.
