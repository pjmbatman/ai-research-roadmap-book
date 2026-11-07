# LoRA의 순전파·병합·파라미터 수

LoRA는 사전학습 가중치 $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$를 고정하고 저랭크 갱신을 더한다.

$$
\mathbf{y} = \mathbf{W}_0\mathbf{x} + \frac{\alpha}{r}\,\mathbf{B}\mathbf{A}\mathbf{x}, \qquad \mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r},\ \mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}.
$$

세 함수를 완성한다. 입력은 batch로 들어온다: `X`의 shape은 `(N, d_in)`, 출력은 `(N, d_out)`이다.

1. `lora_forward(X, W0, A, B, alpha)`: 위 식을 **병합하지 않고** 계산한다. 고정 경로 $\mathbf{X}\mathbf{W}_0^\top$와 저랭크 경로 $(\mathbf{X}\mathbf{A}^\top)\mathbf{B}^\top$를 따로 계산해 더한다. $r$은 `A`의 행 수다.
2. `merge(W0, A, B, alpha)`: 병합한 가중치 $\mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$를 반환한다(shape `(d_out, d_in)`). `W0`를 제자리에서 바꾸지 않는다.
3. `n_trainable(d_in, d_out, r)`: LoRA가 학습하는 파라미터 수(정수)를 반환한다.

테스트는 다음을 확인한다.

- 병합한 가중치의 출력 `X @ merge(...).T`가 `lora_forward`와 같다.
- `B`가 0이면 출력이 사전학습 모델 $\mathbf{X}\mathbf{W}_0^\top$와 같다(학습의 시작점).
- 갱신 $\mathbf{W} - \mathbf{W}_0$의 rank가 $r$ 이하다.
- `merge`가 `W0`를 바꾸지 않는다.
- $1024 \times 1024$ 행렬, $r = 8$이면 학습 파라미터는 16,384개다.
