// 연습문제 채점기. 브라우저 Python에 작은 pytest 호환 계층(approx · raises · mark.parametrize)을 두고
// test_exercise.py의 test_* 함수를 직접 실행해 결과를 JSON으로 돌려준다. 실제 pytest와 같은 테스트 파일을 쓴다.
export const HARNESS = String.raw`
import sys, types, os, json, math, itertools, importlib.util, traceback

def _install_pytest_shim():
    if "pytest" in sys.modules and getattr(sys.modules["pytest"], "__roadmap_shim__", False):
        return
    pt = types.ModuleType("pytest")
    pt.__roadmap_shim__ = True

    class _Approx:
        def __init__(self, expected, rel=None, abs=None):
            self.expected, self.rel, self.abs = expected, rel, abs
        def _close(self, a, b):
            rel = 1e-6 if self.rel is None else self.rel
            ab = 1e-12 if self.abs is None else self.abs
            if self.rel is None and self.abs is not None:
                rel = 0.0
            return abs(a - b) <= max(rel * abs(b), ab)
        def __eq__(self, other):
            try:
                import numpy as np
                a, b = np.asarray(other, dtype=float), np.asarray(self.expected, dtype=float)
                if a.shape != b.shape and a.size != 1 and b.size != 1:
                    return False
                return bool(np.all([self._close(x, y) for x, y in zip(np.broadcast_to(a, np.broadcast(a, b).shape).ravel(), np.broadcast_to(b, np.broadcast(a, b).shape).ravel())]))
            except Exception:
                return self._close(float(other), float(self.expected))
        def __repr__(self):
            return f"approx({self.expected!r})"
    pt.approx = lambda expected, rel=None, abs=None: _Approx(expected, rel, abs)

    class _Raises:
        def __init__(self, exc):
            self.exc = exc
        def __enter__(self):
            return self
        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"{self.exc.__name__} 예외가 발생해야 합니다")
            return issubclass(et, self.exc)
    pt.raises = lambda exc: _Raises(exc)

    class _Mark:
        @staticmethod
        def parametrize(names, values):
            names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
            def deco(fn):
                cases = [dict(zip(names, v if len(names) > 1 else (v,))) for v in values]
                prev = getattr(fn, "_cases", None)
                fn._cases = [{**a, **b} for a, b in itertools.product(prev, cases)] if prev else cases
                return fn
            return deco
    pt.mark = _Mark()
    sys.modules["pytest"] = pt

def run_exercise_tests(folder):
    _install_pytest_shim()
    os.environ["EXERCISE_MODULE"] = "submission"
    results = []
    try:
        spec = importlib.util.spec_from_file_location("roadmap_test_" + os.path.basename(folder).replace("-", "_"), os.path.join(folder, "test_exercise.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return json.dumps([{"name": "코드 불러오기", "ok": False, "msg": f"{type(e).__name__}: {e}\n{tb}"}])
    for name, fn in list(vars(mod).items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        for case in (getattr(fn, "_cases", None) or [{}]):
            label = name + (" [" + ", ".join(f"{k}={v}" for k, v in case.items()) + "]" if case else "")
            try:
                fn(**case)
                results.append({"name": label, "ok": True})
            except AssertionError as e:
                results.append({"name": label, "ok": False, "msg": str(e) or "기대한 값과 다릅니다"})
            except NotImplementedError as e:
                results.append({"name": label, "ok": False, "msg": f"아직 구현하지 않았습니다: {e}"})
            except Exception as e:
                results.append({"name": label, "ok": False, "msg": f"{type(e).__name__}: {e}"})
    return json.dumps(results)
`;
