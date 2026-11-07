# conjugate gradient 구현

대칭 양의 정부호(SPD) 행렬 $\mathbf{A}$에 대한 선형시스템 $\mathbf{A}\mathbf{x} = \mathbf{b}$를 conjugate gradient(CG)로 푼다. CG는 $\mathbf{A}$의 원소가 아니라 **행렬–벡터 곱** $\mathbf{v} \mapsto \mathbf{A}\mathbf{v}$만 쓴다.

$\mathbf{x}_0 = \mathbf{0}$, $\mathbf{r}_0 = \mathbf{b}$, $\mathbf{p}_0 = \mathbf{r}_0$에서 시작해 다음을 반복한다.

$$
\alpha_k = \frac{\mathbf{r}_k^\top\mathbf{r}_k}{\mathbf{p}_k^\top\mathbf{A}\mathbf{p}_k}, \quad
\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k\mathbf{p}_k, \quad
\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k\mathbf{A}\mathbf{p}_k,
$$

$$
\beta_k = \frac{\mathbf{r}_{k+1}^\top\mathbf{r}_{k+1}}{\mathbf{r}_k^\top\mathbf{r}_k}, \quad
\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k\mathbf{p}_k.
$$

함수 `conjugate_gradient(matvec, b, tol=1e-10, max_iter=1000)`를 완성한다.

- `matvec`은 벡터 `v`를 받아 `A @ v`를 돌려주는 함수다. 행렬 자체는 주어지지 않는다.
- 상대 잔차 $\|\mathbf{r}_k\| / \|\mathbf{b}\|$가 `tol` 이하가 되면 멈춘다. 시작할 때 이미 만족하면 반복하지 않는다.
- `(x, n_iter)`를 반환한다. `n_iter`는 수행한 반복(위 갱신) 횟수다.
- `matvec`은 반복마다 **한 번만** 호출한다(테스트가 호출 횟수를 센다). $\mathbf{A}\mathbf{p}_k$를 변수에 담아 $\alpha_k$와 $\mathbf{r}_{k+1}$ 계산에 함께 쓴다.

정확한 산술에서 CG는 $n \times n$ 시스템을 최대 $n$번 안에 푼다. 테스트는 결과의 정확도, 반복 수, `matvec` 호출 수를 확인한다.
