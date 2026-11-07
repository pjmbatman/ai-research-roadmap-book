# 정확히 재개되는 optimizer 만들기

`MomentumSGD`는 선형 회귀 $y \approx X\mathbf{w}$를 minibatch momentum SGD로 학습한다. 매 step은 난수로 minibatch를 뽑고, 10 step마다 학습률을 절반으로 줄이고, 속도(velocity)를 갱신한 뒤 가중치를 움직인다.

$$
\mathbf{v}_{k+1} = \beta\,\mathbf{v}_k + \mathbf{g}_k, \qquad \mathbf{w}_{k+1} = \mathbf{w}_k - \eta_k\,\mathbf{v}_{k+1}, \qquad \eta_k = \eta_0 \cdot 0.5^{\lfloor k/10 \rfloor}
$$

`step()`은 이미 구현되어 있다. 두 메서드를 완성한다.

- `state_dict()`: 재개에 필요한 상태를 dict로 반환한다.
- `load_state_dict(state)`: 그 dict로 상태를 되돌린다.

요구 사항은 하나다. **12 step 학습 → 저장 → 새 객체(다른 시드로 만든)에 복원 → 13 step 학습**의 결과가 **끊김 없이 25 step 학습한 결과와 비트 단위로 같아야 한다.**

- `step()`의 결과에 영향을 주는 값이 무엇인지 코드에서 찾는다. 데이터 `X`, `y`와 하이퍼파라미터는 생성자가 다시 만들어 준다.
- NumPy `Generator`의 상태는 `rng.bit_generator.state`(dict)로 읽고 쓸 수 있다.
- 저장본은 **그 시점의 사진**이어야 한다. 저장한 뒤 학습을 계속해도 저장본이 바뀌면 안 되고, 복원한 객체가 학습해도 저장본이 바뀌면 안 된다. 배열과 dict를 복사한다.
- 저장본은 `pickle`로 직렬화할 수 있어야 한다.
