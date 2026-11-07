class Registry:
    """이름 -> 클래스(또는 함수) 대응표와, 설정 dict에서 객체를 만드는 build."""

    def __init__(self):
        self._items = {}

    def register(self, name):
        """name으로 등록하는 decorator를 반환한다. 중복 이름이면 KeyError."""
        def deco(obj):
            if name in self._items:
                raise KeyError(f"이미 등록된 이름: {name!r}")
            self._items[name] = obj
            return obj
        return deco

    def _resolve(self, value):
        if isinstance(value, dict) and "type" in value:
            return self.build(value)
        if isinstance(value, list):
            return [self._resolve(v) for v in value]
        return value

    def build(self, cfg):
        """cfg["type"]으로 클래스를 찾아 나머지 키를 인자로 호출한다. 중첩 설정도 build한다."""
        kind = cfg["type"]
        if kind not in self._items:
            raise KeyError(f"알 수 없는 type {kind!r}. 등록된 이름: {sorted(self._items)}")
        kwargs = {k: self._resolve(v) for k, v in cfg.items() if k != "type"}
        return self._items[kind](**kwargs)
