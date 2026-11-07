"""6.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-code-engineering/documentation/cells.py
"""
# %% cell 1 — README의 명령이 실제 CLI와 맞는지 확인
import argparse
import re
import shlex

parser = argparse.ArgumentParser(prog="train.py")
parser.add_argument("--config", required=True)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)

readme = """
## 결과 재현
~~~bash
python scripts/train.py --config configs/base.yaml --seed 1
python scripts/train.py --config configs/base.yaml --lr 3e-4 --seed 2
~~~
"""
blocks = re.findall(r"^~~~bash\n(.*?)^~~~", readme, flags=re.S | re.M)
for line in "".join(blocks).splitlines():
    args = shlex.split(line)[2:]                     # "python scripts/train.py" 뒤의 인자
    _, unknown = parser.parse_known_args(args)
    opts = [a for a in unknown if a.startswith("--")]
    print(("OK  " if not opts else f"모르는 옵션 {opts}  ") + line)

# %% cell 2 — shape·단위·예외를 적은 docstring과 실행되는 예제
import doctest
import numpy as np

def heat_flux(temp_k, thickness_m, conductivity_w_mk=0.6):
    """평판을 통과하는 1차원 정상상태 열유속을 계산한다(Fourier 법칙).

    Parameters
    ----------
    temp_k : array_like, shape (..., 2)
        양쪽 면의 온도 [K]. 마지막 축은 (뜨거운 면, 차가운 면).
    thickness_m : float
        판의 두께 [m]. 양수여야 한다.
    conductivity_w_mk : float, optional
        열전도도 [W/(m K)]. 기본값 0.6.

    Returns
    -------
    ndarray, shape (...)
        열유속 [W/m^2]. 뜨거운 면에서 차가운 면으로 흐르면 양수.

    Raises
    ------
    ValueError
        thickness_m이 양수가 아닐 때.

    Examples
    --------
    >>> print(round(float(heat_flux([310.0, 300.0], thickness_m=0.01)), 6))
    600.0
    >>> heat_flux([[310.0, 300.0], [300.0, 310.0]], 0.02).round(6).tolist()
    [300.0, -300.0]
    """
    if not thickness_m > 0:
        raise ValueError(f"thickness_m must be positive, got {thickness_m}")
    t = np.asarray(temp_k, dtype=float)              # (..., 2)
    return conductivity_w_mk * (t[..., 0] - t[..., 1]) / thickness_m

runner = doctest.DocTestRunner()
for test in doctest.DocTestFinder().find(heat_flux, globs={"heat_flux": heat_flux}):
    print(runner.run(test))

# %% cell 3 — 코드만 바뀌고 문서는 그대로일 때
class Report(doctest.DocTestRunner):
    def report_failure(self, out, test, example, got):
        out(f"실패: {example.source.strip()}\n  문서: {example.want.strip()}\n  실제: {got.strip()}\n")

def heat_flux_v2(temp_k, thickness_m, conductivity_w_mk=0.58):   # 기본값만 바꾼 새 버전
    t = np.asarray(temp_k, dtype=float)
    return conductivity_w_mk * (t[..., 0] - t[..., 1]) / thickness_m

old_doc = doctest.DocTestParser().get_doctest(
    heat_flux.__doc__, {"heat_flux": heat_flux_v2, "np": np}, "heat_flux", None, 0)
print(Report().run(old_doc))
print("np.mean의 repr:", repr(np.mean([1.0, 2.0])))   # 문서 예제에 그대로 쓰면 NumPy 버전에 따라 깨진다
