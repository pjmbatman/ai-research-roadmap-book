# 오타를 잡는 설정 깊은 병합

기본 설정 위에 실험 파일과 명령행 override를 차례로 덮어쓰려면 중첩 dict를 **깊게** 병합해야 한다. `{**base, **override}`는 최상위 키만 바꾸므로, `{"model": {"width": 128}}`을 덮어쓰면 `model` 아래의 다른 키가 모두 사라진다.

병합에는 또 하나의 함정이 있다. `train.learning_rate`처럼 기본 설정에 없는 키를 덮어쓰면, 대부분의 병합 함수는 새 키를 조용히 추가한다. 코드는 여전히 `train.lr`을 읽으므로 실험자가 바꿨다고 믿은 값은 아무 효과가 없다.

함수 `deep_merge(base, override)`를 완성한다.

- `override`의 값이 dict이고 `base`의 같은 키도 dict이면 재귀적으로 병합한다.
- 그 밖에는 `override`의 값이 `base`의 값을 대체한다. 리스트도 이어 붙이지 않고 통째로 대체한다.
- `override`에 `base`에 없는 키가 있으면 `KeyError`를 낸다. 메시지에 점 경로(예: `train.learning_rate`)를 넣는다.
- 한쪽은 dict이고 다른 쪽은 dict가 아니면 `TypeError`를 낸다(예: `model`이 섹션인데 `model=5`로 덮어씀). 메시지에 점 경로를 넣는다.
- 새 dict를 반환한다. `base`와 `override`를 바꾸지 않고, 반환한 dict를 수정해도 입력이 바뀌지 않아야 한다(중첩된 dict·list를 공유하지 않는다).

```python
base = {"model": {"width": 64, "depth": 3}, "train": {"lr": 1e-3}}
deep_merge(base, {"model": {"width": 128}})
# {'model': {'width': 128, 'depth': 3}, 'train': {'lr': 0.001}}
deep_merge(base, {"train": {"learning_rate": 3e-4}})
# KeyError: 'train.learning_rate'
```

힌트: 재귀 호출에 지금까지의 경로를 넘기면 오류 메시지에 점 경로를 쓸 수 있다. 공유를 막으려면 `copy.deepcopy`를 쓴다.
