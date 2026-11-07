# state_dict 비교와 접두사 정리

체크포인트를 불러올 때 가장 흔한 문제는 **이름 불일치**다. 다른 코드가 저장한 state_dict에는 `module.`이나 `model.` 같은 접두사가 붙어 있거나, 층 이름이 바뀌었거나, shape이 다른 텐서가 섞여 있다. `strict=False`로 그냥 불러오면 오류 없이 일부만 로드되므로, 로드 전에 무엇이 어긋나는지 먼저 확인해야 한다.

state_dict는 이름(문자열)에서 배열로 가는 dict다. 이 문제에서는 PyTorch 대신 NumPy 배열을 쓴다. 두 함수를 완성한다.

1. `strip_prefix(state_dict, prefix)` — `prefix`로 **시작하는** 키에서만 접두사를 떼어 낸 새 dict를 반환한다. 다른 키는 그대로 둔다. 원본 dict를 바꾸지 않는다. 예: `{"module.fc.weight": W}`, `"module."` → `{"fc.weight": W}`.
2. `diff_state_dict(expected_shapes, state_dict)` — 모델이 기대하는 `{이름: shape}`과 불러온 state_dict를 비교해 다음 세 목록을 담은 dict를 반환한다. 각 목록은 이름을 **정렬**한 list다.
   - `"missing"`: 모델에는 있는데 state_dict에 없는 이름
   - `"unexpected"`: state_dict에는 있는데 모델에 없는 이름
   - `"shape_mismatch"`: 양쪽에 다 있지만 shape이 다른 이름

```python
diff_state_dict({"fc.weight": (2, 3), "fc.bias": (2,)},
                {"fc.weight": np.zeros((3, 3)), "head.bias": np.zeros(1)})
# {"missing": ["fc.bias"], "unexpected": ["head.bias"], "shape_mismatch": ["fc.weight"]}
```

세 목록이 모두 비어 있어야 `strict=True` 로드가 성공한다.
