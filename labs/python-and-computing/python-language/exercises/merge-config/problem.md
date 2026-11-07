# 원본을 건드리지 않는 설정 병합

실험 설정은 흔히 중첩 dict로 둔다. 기본 설정 `base`에 실험별 변경 사항 `overrides`를 덮어써 새 설정을 만드는 함수 `merge_config(base, overrides)`를 완성한다.

```python
base = {"model": {"width": 64, "depth": 2}, "tags": ["baseline"]}
overrides = {"model": {"depth": 4}, "seed": 1}
merge_config(base, overrides)
# {"model": {"width": 64, "depth": 4}, "tags": ["baseline"], "seed": 1}
```

규칙은 다음과 같다.

- 같은 키의 값이 **둘 다 dict**이면 재귀적으로 병합한다.
- 그 밖의 경우(값이 dict가 아니거나 한쪽만 dict)에는 `overrides`의 값이 이긴다.
- `base`와 `overrides`는 **어떤 깊이에서도 바뀌면 안 된다.**
- 반환한 설정은 입력과 **객체를 공유하지 않아야** 한다. 반환값의 리스트에 원소를 추가해도 `base`의 리스트가 바뀌면 안 된다.

테스트는 결과 값뿐 아니라, 병합 뒤 입력이 그대로인지와 결과를 수정했을 때 입력이 영향을 받지 않는지를 확인한다.

힌트: `dict(base)`나 `base.copy()`는 얕은 복사다. 어느 단계에서 깊은 복사가 필요한지 생각한다.
