def resolve(name: str, package: str | None) -> str:
    """상대 import 이름 name을 package(__package__) 기준의 절대 모듈 이름으로 바꾼다.

    점이 없으면 name을 그대로 반환한다.
    package가 None이거나 비었거나, 최상위보다 위로 올라가면 ImportError.
    """
    if not name.startswith("."):
        return name                                   # 절대 import
    if not package:
        raise ImportError(f"'{name}': 기준 패키지가 없어 상대 import를 풀 수 없습니다")
    level = len(name) - len(name.lstrip("."))
    rest = name[level:]
    parts = package.split(".")
    if level - 1 >= len(parts):
        raise ImportError(f"'{name}': 최상위 패키지 '{parts[0]}'보다 위로 올라갑니다")
    base = ".".join(parts[: len(parts) - (level - 1)])
    return f"{base}.{rest}" if rest else base
