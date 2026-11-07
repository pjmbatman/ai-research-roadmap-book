import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


doc_drift = load().doc_drift


def in_sync(temp_k, thickness_m, conductivity_w_mk=0.6):
    """열유속을 계산한다.

    Parameters
    ----------
    temp_k : array_like, shape (..., 2)
        양쪽 면의 온도 [K].
    thickness_m : float
        판의 두께 [m].
    conductivity_w_mk : float, optional
        열전도도 [W/(m K)].

    Returns
    -------
    flux : ndarray
        열유속 [W/m^2].
    """


def renamed(temp_k, thickness_mm, conductivity_w_mk=0.6):
    """두께 단위를 mm로 바꿨지만 문서는 그대로다.

    Parameters
    ----------
    temp_k : array_like
        온도 [K].
    thickness_m : float
        두께 [m].
    conductivity_w_mk : float
        열전도도 [W/(m K)].
    """


def combined_and_star(x, y, *args, scale=1.0, **kwargs):
    """여러 이름을 한 줄에 적고, 별표 인자를 문서화한다.

    Parameters
    ----------
    x, y : ndarray, shape (N,)
        좌표 [m]. 예: x : 동쪽 방향
    *args : float
        추가 계수.
    **kwargs
        그 밖의 설정.

    Raises
    ------
    ValueError
        x와 y의 길이가 다를 때.
    """


def no_doc(a, b):
    pass


def no_parameters_section(a, b=2):
    """설명만 있고 Parameters 절이 없다.

    Returns
    -------
    a : int
    """


class Model:
    def fit(self, data, epochs=10):
        """학습한다.

        Parameters
        ----------
        data : ndarray
            학습 데이터.
        lr : float
            학습률. (이제는 생성자에서 받는다)
        """

    @classmethod
    def load(cls, path):
        """불러온다.

        Parameters
        ----------
        path : str
            checkpoint 경로.
        """


def test_in_sync_ignores_returns_section():
    assert doc_drift(in_sync) == {"missing": [], "stale": []}, "Returns 절의 'flux'는 인자가 아닙니다"


def test_renamed_argument():
    assert doc_drift(renamed) == {"missing": ["thickness_mm"], "stale": ["thickness_m"]}


def test_combined_names_stars_and_description_colons():
    r = doc_drift(combined_and_star)
    assert r == {"missing": ["scale"], "stale": []}, (
        f"x, y는 두 이름, *args·**kwargs는 별표를 떼고, 들여쓴 설명 줄의 ':'은 무시합니다. 현재 {r}")


def test_no_docstring():
    assert doc_drift(no_doc) == {"missing": ["a", "b"], "stale": []}


def test_no_parameters_section():
    assert doc_drift(no_parameters_section) == {"missing": ["a", "b"], "stale": []}, \
        "Returns 절만 있으면 문서화된 인자는 없습니다"


def test_methods_skip_self_and_cls():
    assert doc_drift(Model().fit) == {"missing": ["epochs"], "stale": ["lr"]}
    assert doc_drift(Model.fit) == {"missing": ["epochs"], "stale": ["lr"]}, "self는 제외합니다"
    assert doc_drift(Model.load) == {"missing": [], "stale": []}


def test_lists_are_sorted():
    def f(zeta, alpha, mid):
        """
        Parameters
        ----------
        omega : int
        beta : int
        """
    assert doc_drift(f) == {"missing": ["alpha", "mid", "zeta"], "stale": ["beta", "omega"]}
