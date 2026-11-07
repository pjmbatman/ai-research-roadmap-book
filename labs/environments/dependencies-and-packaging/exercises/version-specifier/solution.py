OPERATORS = ("~=", "==", "!=", ">=", "<=", ">", "<")   # 두 글자 연산자를 먼저 본다


def _v(s):
    return tuple(int(p) for p in s.strip().split("."))   # "1.26.4" -> (1, 26, 4)


def _pad(a, b):
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


def _clause(version, clause):
    op = next(o for o in OPERATORS if clause.startswith(o))
    target = clause[len(op):].strip()
    v = _v(version)
    if target.endswith(".*"):                            # ==1.26.* / !=1.26.*: 앞자리 비교
        prefix = _v(target[:-2])
        same = _pad(v, prefix)[0][: len(prefix)] == prefix
        return same if op == "==" else not same
    t = _v(target)
    if op == "~=":                                       # ~=1.4.2  ==  >=1.4.2, ==1.4.*
        return _clause(version, ">=" + target) and _clause(version, "==" + ".".join(map(str, t[:-1])) + ".*")
    a, b = _pad(v, t)
    return {"==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b}[op]


def satisfies(version, spec):
    """version(예 "1.26.4")이 spec(예 ">=1.23,<2.6")의 모든 조건을 만족하면 True.

    지원: ==, !=, >=, <=, >, <, ~=, 그리고 ==X.Y.* / !=X.Y.*
    버전은 점으로 구분한 정수만(사전 배포 rc·dev 등은 다루지 않는다).
    """
    return all(_clause(version, c.strip()) for c in spec.split(",") if c.strip())
