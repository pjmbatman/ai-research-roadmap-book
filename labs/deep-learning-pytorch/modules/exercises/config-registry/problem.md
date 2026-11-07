# 설정에서 객체 만들기: registry 패턴

실험마다 모델 코드를 고치는 대신, 설정 파일(TOML·YAML)에 `type`과 인자를 적고 코드가 그 설정에서 객체를 만들게 하면 실험의 차이가 설정 파일의 diff로 남는다. 이 문제에서는 PyTorch 없이 그 뼈대를 만든다.

`Registry` 클래스의 두 메서드를 완성한다.

1. `register(name)` — 클래스(또는 함수)를 `name`으로 등록하는 **decorator**를 반환한다. decorator는 받은 객체를 그대로 돌려준다. 이미 등록된 이름이면 `KeyError`를 낸다(두 모듈이 같은 이름을 쓰는 사고를 막는다).
2. `build(cfg)` — dict `cfg`에서 객체를 만든다.
   - `cfg["type"]`으로 등록된 클래스를 찾고, 나머지 키를 키워드 인자로 넘겨 호출한다.
   - 등록되지 않은 `type`이면 `KeyError`를 내고, 메시지에 **등록된 이름들을 정렬해** 포함한다(오타를 바로 찾을 수 있게).
   - 인자 값이 `"type"` 키를 가진 dict이면 그 값도 먼저 `build`한다. 인자 값이 list이면 원소마다 같은 규칙을 적용한다(중첩 설정).
   - 받은 `cfg`를 바꾸지 않는다. 같은 설정으로 `build`를 두 번 호출해도 결과가 같아야 한다.

```python
reg = Registry()

@reg.register("linear")
class Linear:
    def __init__(self, d_in, d_out): ...

@reg.register("seq")
class Seq:
    def __init__(self, layers): ...

model = reg.build({"type": "seq", "layers": [{"type": "linear", "d_in": 3, "d_out": 8},
                                              {"type": "linear", "d_in": 8, "d_out": 1}]})
```
