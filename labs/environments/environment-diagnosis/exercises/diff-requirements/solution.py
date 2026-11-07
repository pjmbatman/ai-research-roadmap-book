import re


def _norm(name):
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _parse(text):
    pkgs = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" in line:
            name, version = line.split("==", 1)
        elif "@" in line:
            name, version = line.split("@", 1)
        else:
            continue
        pkgs[_norm(name)] = version.strip()
    return pkgs


def diff_requirements(before, after):
    """freeze 형식 문자열 두 개를 비교한다.

    before, after: "numpy==2.1.3\\npandas==2.2.3\\n..." 형식의 문자열
    반환: {"added": {이름: 버전}, "removed": {이름: 버전}, "changed": {이름: (옛, 새)}}
    """
    a, b = _parse(before), _parse(after)
    return {
        "added": {k: b[k] for k in b.keys() - a.keys()},
        "removed": {k: a[k] for k in a.keys() - b.keys()},
        "changed": {k: (a[k], b[k]) for k in a.keys() & b.keys() if a[k] != b[k]},
    }
