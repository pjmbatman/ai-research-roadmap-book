import inspect


def _is_underline(line):
    s = line.strip()
    return len(s) >= 3 and set(s) == {"-"}


def _documented(doc):
    lines = doc.splitlines()
    start = None
    for i in range(len(lines) - 1):
        if lines[i].strip() == "Parameters" and _is_underline(lines[i + 1]):
            start = i + 2
            break
    if start is None:
        return set()
    names = set()
    for i in range(start, len(lines)):
        line = lines[i]
        if i + 1 < len(lines) and line.strip() and _is_underline(lines[i + 1]):
            break                                    # 다음 절의 제목
        if not line.strip() or line[0].isspace():
            continue                                 # 빈 줄이나 설명 줄
        head = line.split(":", 1)[0]
        for name in head.split(","):
            name = name.strip().lstrip("*")
            if name:
                names.add(name)
    return names


def doc_drift(func):
    """NumPy 스타일 docstring의 Parameters 절과 func의 signature를 비교한다.

    반환: {"missing": signature에만 있는 인자, "stale": 문서에만 있는 인자}, 각각 정렬된 리스트
    """
    params = {n for n in inspect.signature(func).parameters if n not in ("self", "cls")}
    doc = inspect.getdoc(func) or ""
    documented = _documented(doc)
    return {"missing": sorted(params - documented), "stale": sorted(documented - params)}
