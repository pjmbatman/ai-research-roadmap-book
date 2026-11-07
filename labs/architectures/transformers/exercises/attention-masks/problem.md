# causal mask와 padding mask 만들기

길이가 다른 시퀀스 $B$개를 PAD로 채워 길이 $T$인 batch를 만들었다. 시퀀스 $b$의 실제 길이는 `lengths[b]`이고, 위치 `lengths[b]`부터 `T-1`까지는 PAD다. self-attention에 쓸 bool mask `(B, T, T)`를 만든다. `mask[b, t, s]`가 `True`이면 시퀀스 $b$의 query 위치 $t$가 key 위치 $s$를 볼 수 있다.

함수 `make_mask(lengths, T, causal)`을 완성한다.

- **padding mask**: key 위치 $s$가 PAD이면($s \ge$ `lengths[b]`) 어떤 query도 볼 수 없다.
- **causal mask**(`causal=True`일 때만): query 위치 $t$는 $s \le t$인 key만 볼 수 있다.
- 두 조건을 모두 만족하는 칸만 `True`다.

PAD 위치의 **query** 행은 이 문제에서 따로 가리지 않는다(key 쪽 조건만 적용). 그 행의 출력은 손실에서 빼면 된다. 다만 그 행이 모두 `False`가 되면 softmax에서 `nan`이 나오므로, 테스트는 어떤 행도 모두 `False`가 아닌지 확인한다(실제 길이가 1 이상이면 위 규칙으로 자동으로 만족된다).

테스트는 다음을 확인한다.

- shape이 `(B, T, T)`이고 dtype이 bool이다.
- `causal=False`이면 PAD key 열만 `False`다.
- `causal=True`이면 실제 길이 안에서 아래삼각(대각 포함)만 `True`다.
- 이 mask로 attention을 계산하면, PAD 칸의 값을 바꿔도 실제 토큰의 출력이 변하지 않고, causal인 경우 미래 토큰을 바꿔도 과거 출력이 변하지 않는다.

힌트: `np.arange(T)`와 broadcasting으로 `(B, 1, T)` 모양의 key 조건과 `(1, T, T)` 모양의 causal 조건을 만들어 `&`로 합친다.
