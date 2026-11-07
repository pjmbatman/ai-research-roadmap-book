class Registry:
    """이름 -> 클래스(또는 함수) 대응표와, 설정 dict에서 객체를 만드는 build."""

    def __init__(self):
        self._items = {}

    def register(self, name):
        """name으로 등록하는 decorator를 반환한다. 중복 이름이면 KeyError."""
        raise NotImplementedError("decorator를 반환하세요")

    def build(self, cfg):
        """cfg["type"]으로 클래스를 찾아 나머지 키를 인자로 호출한다. 중첩 설정도 build한다."""
        raise NotImplementedError("type을 찾고, 인자를 재귀적으로 만든 뒤 호출하세요")
