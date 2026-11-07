"""17.10 실습: 같은 작은 MLP를 PyTorch와 JAX로 학습해 코드 구조·결과·속도를 비교한다.

jax는 실습 공통 환경에 들어 있지 않다. uv의 --with로 이 실행에만 더한다(집필 시점 예시).

    uv run --project labs --group torch --with jax python labs/deep-learning-pytorch/pytorch-ecosystem/compare_torch_jax.py

두 구현은 NumPy로 만든 같은 초기 가중치와 같은 데이터에서 시작해, full-batch gradient descent로 같은 step 수만큼 학습한다.
시간은 기계와 부하에 따라 달라진다. 숫자 자체보다 '처음 호출(컴파일 포함)'과 '이후 호출'의 차이를 본다.
"""
import time

import numpy as np

STEPS, LR = 500, 0.1
rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(256, 2)).astype(np.float32)                     # (N, 2)
Y = (np.sin(2 * X[:, :1]) * np.cos(X[:, 1:])).astype(np.float32)             # (N, 1)
INIT = {"W1": rng.normal(0, 0.7, (2, 32)), "b1": np.zeros(32), "W2": rng.normal(0, 0.18, (32, 1)), "b2": np.zeros(1)}
INIT = {k: v.astype(np.float32) for k, v in INIT.items()}


def run_torch():
    import torch
    torch.set_num_threads(1)
    p = {k: torch.tensor(v, requires_grad=True) for k, v in INIT.items()}    # 파라미터를 직접 들고 있다
    x, y = torch.from_numpy(X), torch.from_numpy(Y)
    times = []
    for _ in range(STEPS):
        t = time.perf_counter()
        pred = torch.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        loss = ((pred - y) ** 2).mean()
        grads = torch.autograd.grad(loss, list(p.values()))                   # 상태를 가진 그래프에서 역전파
        with torch.no_grad():
            for v, g in zip(p.values(), grads):
                v -= LR * g                                                  # 제자리(in-place) 업데이트
        times.append(time.perf_counter() - t)
    return loss.item(), times


def run_jax():
    import jax
    import jax.numpy as jnp

    def loss_fn(p, x, y):                                                    # 순수 함수: 파라미터를 인자로 받는다
        pred = jnp.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        return jnp.mean((pred - y) ** 2)

    @jax.jit                                                                 # 한 step 전체를 컴파일
    def step(p, x, y):
        loss, g = jax.value_and_grad(loss_fn)(p, x, y)
        return jax.tree_util.tree_map(lambda v, gv: v - LR * gv, p, g), loss  # 새 파라미터를 반환

    p = {k: jnp.asarray(v) for k, v in INIT.items()}
    times = []
    for _ in range(STEPS):
        t = time.perf_counter()
        p, loss = step(p, X, Y)
        loss.block_until_ready()                                             # 비동기 실행을 기다려야 시간이 맞다
        times.append(time.perf_counter() - t)
    return float(loss), times


for name, fn in [("PyTorch (eager)", run_torch), ("JAX (jit)", run_jax)]:
    loss, times = fn()
    print(f"{name:16s} 최종 손실 {loss:.6f}  첫 step {times[0] * 1e3:8.1f} ms  이후 step 중앙값 {np.median(times[1:]) * 1e3:6.3f} ms")
