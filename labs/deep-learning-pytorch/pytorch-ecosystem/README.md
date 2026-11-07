# 17.10 PyTorch 생태계와 대안 — 실습

장: [content/17-deep-learning-pytorch/10-pytorch-ecosystem](../../../content/17-deep-learning-pytorch/10-pytorch-ecosystem/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀(`torch.compile`의 그래프 세기, `torch.func`, 미분가능 시뮬레이션). PyTorch 필요 |
| `compare_torch_jax.py` | 같은 MLP를 PyTorch eager와 JAX jit으로 학습해 결과·시간 비교 |
| `test_quiz.py` | 퀴즈 정답 검증 |

jax는 실습 공통 환경(`labs/pyproject.toml`)에 들어 있지 않다. `uv run --with jax`로 그 실행에만 더한다(집필 시점 예시).

```bash
# 저장소 루트에서
uv run --project labs --group torch python labs/deep-learning-pytorch/pytorch-ecosystem/cells.py
uv run --project labs --group torch --with jax python labs/deep-learning-pytorch/pytorch-ecosystem/compare_torch_jax.py
```

기대 결과: 두 프레임워크의 최종 손실이 소수 여섯째 자리까지 같다. 시간은 기계마다 다르며, JAX의 첫 step에는 컴파일 시간이 들어간다.
