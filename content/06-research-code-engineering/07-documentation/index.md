---
id: research-code-engineering/documentation
title: "연구 코드 문서화"
title_en: "Documenting Research Code"
tier: core
status: draft
prerequisites: [research-code-engineering/project-structure]
---

# 연구 코드 문서화

연구 코드의 첫 독자는 6개월 뒤의 자신이다. 그때의 자신은 어떤 데이터 파일을 어디에 두었는지, 어떤 명령으로 논문의 표 2를 만들었는지, 왜 시료 단위로 데이터를 나눴는지 기억하지 못한다. 두 번째 독자는 프로젝트를 이어받는 동료이고, 세 번째 독자는 논문을 읽고 결과를 재현하려는 다른 연구실 사람이다. 셋 모두 작성자에게 물어볼 수 없거나, 물어보기 어렵다.

이 장은 연구 코드의 문서를 네 층으로 나눈다. 저장소에 처음 온 사람을 설치부터 재현까지 안내하는 **README**, 함수 하나를 바르게 쓰게 하는 **docstring**, 코드가 말하지 못하는 "무엇을, 왜"를 남기는 **설계 문서**, 그리고 결과가 언제 왜 바뀌었는지 남기는 **변경 기록**이다. 각 층의 공통 원칙은 하나다. 문서는 코드와 함께 낡으므로, **가능한 한 실행해서 확인할 수 있게** 쓴다. 도구 이름과 명령은 집필 시점의 예시다.

## 학습 목표

- README로 설치부터 재현까지 안내한다
- docstring과 설계 문서를 작성한다

## README

README는 저장소의 첫 화면이다. GitHub 같은 호스팅 서비스는 저장소 루트의 `README.md`를 첫 페이지에 보여 준다. 연구 프로젝트의 README가 답해야 하는 질문은 순서대로 여섯 가지다.

| 절 | 답하는 질문 | 쓰는 것 |
|---|---|---|
| 목적 | 이것은 무엇이고 무엇을 하지 않는가 | 한두 문단. 대응하는 논문·보고서, 현재 상태(진행 중, 논문 제출본 고정 등) |
| 설치 | 어떻게 같은 환경을 만드는가 | lockfile로 설치하는 정확한 명령, Python·CUDA 등 시스템 요구사항, 확인 명령 |
| 데이터 준비 | 데이터는 어디서 받아 어디에 두는가 | 내려받기 스크립트나 절차, 놓을 경로, 크기, 접근 권한이 필요한지, 무결성 확인(checksum) |
| 실행 | 어떻게 돌리는가 | 학습·평가·예측의 대표 명령, 설정 파일의 위치(6.3장), CLI 도움말(6.4장) |
| 결과 재현 | 논문의 숫자를 어떻게 다시 만드는가 | 표·그림마다 명령, **기대 결과**(숫자와 허용 범위), 대략의 실행 시간과 필요한 자원 |
| 인용 | 이 코드를 쓴 사람은 무엇을 인용하는가 | 논문 서지 정보, `CITATION.cff`, 라이선스 |

가장 자주 빠지는 것은 **기대 결과**다. "`python scripts/eval.py --config configs/paper.yaml`을 실행한다"만 있으면, 따라 한 사람은 출력이 0.912일 때 성공했는지 실패했는지 알 수 없다. "test MAE 0.031 ± 0.002 K(시드 5개 평균), GPU 1장에서 약 20분"이라고 적혀 있으면 스스로 판단할 수 있다. 결과가 시드에 따라 흔들리면 그 범위를 함께 적는다.

연구 프로젝트 README의 뼈대는 다음과 같다. 6.1장의 디렉터리 구조를 전제로 한다.

````markdown
# thermal-surrogate

평판 열전달 시뮬레이션을 대신하는 신경망 surrogate 모델. 논문 "..."(2026)의 코드다.
논문 제출본은 태그 `v1.0`에 고정되어 있다.

## 설치
Python 3.12, uv가 필요하다(집필 시점 예시: `uv --version`으로 확인).
```bash
git clone <저장소 주소> && cd thermal-surrogate
uv sync --locked                 # uv.lock 그대로 설치
uv run python -m thermal_surrogate.check_env   # 환경 확인: "ok"가 출력되어야 한다
```

## 데이터 준비
```bash
uv run python scripts/download_data.py --out data/raw   # 약 2.1 GB, sha256 검사 포함
```
`data/raw/sim_*.h5` 파일 48개가 생긴다. 데이터 설명서: docs/data.md

## 실행
```bash
uv run python scripts/train.py --config configs/base.yaml --seed 0
uv run python scripts/eval.py --run outputs/<run-id>
```

## 결과 재현
| 결과 | 명령 | 기대 결과 | 시간 |
|---|---|---|---|
| 표 2, 1행 | `bash scripts/reproduce_table2.sh` | test MAE 0.031 ± 0.002 K | GPU 1장 약 20분 |

## 인용
CITATION.cff를 참고한다. 라이선스: MIT
````

이 뼈대에서 몇 가지 원칙이 보인다.

- **명령은 복사해서 그대로 실행되게 쓴다.** "환경을 만든 뒤 학습을 돌린다" 대신 실제 명령을 쓴다. 사용자마다 다른 부분은 `<run-id>`처럼 꺾쇠로 표시하고, 그 값을 어디서 얻는지 적는다.
- **각 단계 끝에 확인 방법을 둔다.** "ok가 출력되어야 한다", "파일 48개가 생긴다". 문제가 어느 단계에서 생겼는지 독자가 스스로 알 수 있다.
- **README는 입구이지 전부가 아니다.** 데이터의 자세한 설명, 모델 구조, 결정의 이유는 `docs/`의 별도 문서에 두고 링크한다. README가 길어지면 설치 명령이 스크롤 아래로 밀려 읽히지 않는다.
- **인용 정보는 기계가 읽을 수 있게.** 저장소 루트에 `CITATION.cff`(Citation File Format) 파일을 두면 GitHub가 저장소 화면에 인용 정보를 표시한다.

README의 명령이 실제로 동작하는지는 사람의 주의로 지키기 어렵다. 코드를 바꾼 사람은 README를 떠올리지 못하고, 틀린 문서는 오류를 내지 않는다. 그래서 README의 명령을 **검사 가능한 대상**으로 만든다. 가장 강한 방법은 CI에서 README의 코드 블록을 뽑아 아주 작은 설정으로 실제로 실행하는 것이다. 그보다 가벼운 방법으로, 명령의 인자가 실제 CLI parser(6.4장)에 있는지만 확인할 수도 있다.

```python {exec="browser" session="docs"}
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
```

```output
OK  python scripts/train.py --config configs/base.yaml --seed 1
모르는 옵션 ['--lr']  python scripts/train.py --config configs/base.yaml --lr 3e-4 --seed 2
```

CLI의 인자 이름이 `--lr`에서 `--learning-rate`로 바뀌었는데 README의 두 번째 명령은 그대로였다. 이 검사를 CI에 넣으면 인자 이름을 바꾼 PR이 바로 실패하고, 그 PR에서 README도 함께 고치게 된다. (예에서는 마크다운 코드 블록의 다른 표기인 `~~~`를 썼다. 실제 README에서는 ```` ``` ````로 쓴 블록도 함께 찾는다.)

::quiz{id="readme-essentials"}

:::note[분야 예시]
기후 모델의 출력을 후처리하는 프로젝트는 데이터 준비 절이 README의 대부분을 차지하기도 한다. 어느 모델·시나리오·변수·기간을 받는지, 데이터 포털의 계정이나 동의가 필요한지, 받은 뒤의 전체 용량, 격자를 맞추는(regridding) 단계가 있는지를 적는다. 데이터 제공처가 파일을 갱신하면 같은 이름의 파일이라도 내용이 바뀔 수 있으므로, 받은 날짜와 파일 버전 또는 checksum을 함께 기록해 두는 것이 재현의 조건이 된다.
:::

::quiz{id="readme-command-drift"}

## docstring

README가 저장소의 입구라면 docstring은 함수의 사용 설명서다. 함수 정의 바로 아래의 문자열 리터럴이 docstring이 되고, `help(함수)`와 IDE의 툴팁, 문서 생성 도구가 이것을 보여 준다. 형식에 대한 Python의 기본 관례는 PEP 257에 있다. 연구 코드에서 docstring이 특히 중요한 이유는, 함수를 잘못 쓰는 가장 흔한 원인인 **shape과 단위**가 signature에 드러나지 않기 때문이다. `def heat_flux(temp, thickness)`만 보고는 온도가 켈빈인지 섭씨인지, 두께가 m인지 mm인지, `temp`가 어떤 shape이어야 하는지 알 수 없다.

과학 Python 생태계(NumPy, SciPy, scikit-learn 등)에서 널리 쓰는 형식은 **NumPy 스타일**이다. 절 제목 아래에 `-`로 밑줄을 긋고, 인자마다 `이름 : 타입` 줄과 들여쓴 설명을 쓴다. 다른 흔한 형식으로 `Args:`, `Returns:`를 쓰는 Google 스타일이 있다. 어느 쪽이든 프로젝트 안에서 하나로 통일하면 된다(6.2장). 연구 함수의 docstring에 들어갈 것은 다음과 같다.

- **한 줄 요약**: 함수가 하는 일을 동사로. 이름을 반복하지 않는다.
- **인자**(Parameters): 타입, **shape**(축의 의미 포함), **단위**, 허용 범위, 기본값의 의미.
- **반환값**(Returns): 타입, shape, 단위, 부호 규약.
- **예외**(Raises): 어떤 입력에서 어떤 예외를 내는가(6.5장).
- **예제**(Examples): 실행할 수 있는 짧은 예.
- 필요하면 **근거**(Notes, References): 공식, 출처, 수치적 주의점.

예제는 대화형 셸의 `>>>` 형식으로 쓴다. Python 표준 라이브러리의 `doctest`는 docstring에서 `>>>` 줄을 찾아 실행하고, 출력이 적힌 기대 출력과 **문자열로** 같은지 확인한다. 즉 docstring의 예제가 그대로 테스트가 된다.

```python {exec="browser" session="docs"}
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
```

```output
TestResults(failed=0, attempted=2)
```

식으로 쓰면 $q = k\,(T_\text{hot} - T_\text{cold}) / L$이다. $q$는 열유속[W/m²], $k$는 열전도도[W/(m·K)], $T_\text{hot}, T_\text{cold}$는 두 면의 온도[K], $L$은 두께[m]다. 첫 예제는 $0.6 \times 10 / 0.01 = 600$이다. docstring의 두 예제가 실행되어 모두 통과했다. 예제에서 결과를 `round`하고 `print`나 `.tolist()`로 출력 모양을 고정한 데는 이유가 있다. doctest는 문자열을 비교하므로, 부동소수의 마지막 자리 차이(`599.9999999999999`)나 출력 표현의 차이도 실패가 된다. 특히 NumPy 2.0부터는 NumPy 스칼라의 repr이 타입을 드러내는 형태로 바뀌어, 예제에 `np.mean(...)`의 결과를 그대로 적으면 NumPy 버전에 따라 깨진다.

이제 누군가 기본 열전도도를 0.58로 고치고 docstring은 그대로 두었다고 하자. 같은 docstring의 예제를 새 코드에 실행한다.

```python {exec="browser" session="docs"}
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
```

```output
실패: print(round(float(heat_flux([310.0, 300.0], thickness_m=0.01)), 6))
  문서: 600.0
  실제: 580.0
실패: heat_flux([[310.0, 300.0], [300.0, 310.0]], 0.02).round(6).tolist()
  문서: [300.0, -300.0]
  실제: [290.0, -290.0]
TestResults(failed=2, attempted=2)
np.mean의 repr: np.float64(1.5)
```

두 예제가 모두 실패해 문서가 낡았다는 것을 알려 준다. 실제 프로젝트에서는 `python -m doctest 모듈.py -v`로 모듈 하나를, `pytest --doctest-modules`로 프로젝트 전체의 docstring 예제를 테스트와 함께 실행한다(6.6장). 실행되는 예제는 문서와 코드가 어긋나는 것을 막는 가장 싼 장치다. 물론 doctest가 잡는 것은 예제가 있는 부분뿐이다. 인자 이름을 바꾸고 `Parameters` 절을 고치지 않은 것은 잡지 못한다. 이것은 연습문제에서 signature와 docstring을 비교하는 검사기로 다룬다.

docstring에서 하지 않을 것도 있다. 코드를 문장으로 옮겨 적지 않는다("x에 2를 곱한다"). 작성자·날짜·수정 이력은 git이 정확히 기록하므로 쓰지 않는다. 함수 안의 주석은 **왜** 그렇게 했는지(수치 안정성 때문에 최댓값을 뺀다, 데이터의 결측 표기가 -9999다)를 쓰는 곳이다. 무엇을 하는지는 코드와 이름이 말해야 한다.

::quiz{id="docstring-research-essentials"}

**문서 사이트.** 프로젝트가 커지면 docstring을 모아 웹 문서로 만든다. Sphinx(`sphinx.ext.napoleon` 확장이 NumPy·Google 스타일을 읽는다)나 MkDocs와 mkdocstrings 같은 도구가 흔히 쓰인다(집필 시점 예시). 연구 프로젝트 대부분에는 README와 `docs/`의 Markdown 몇 개로 충분하다. 도구를 들이기 전에 문서의 내용부터 갖춘다.

::quiz{id="doctest-numpy-repr"}

## 설계 문서

코드는 **무엇을** 하는지 말하지만, **무엇에 대해** 하는지(데이터), **왜** 그렇게 하는지(결정)는 말하지 않는다. 연구 프로젝트에는 이것을 남기는 짧은 문서 네 가지가 유용하다. 모두 저장소의 `docs/`에 Markdown으로 두고 코드와 함께 버전 관리한다.

**데이터 설명서**(data description, datasheet). 데이터의 출처와 수집 방법, 변수마다 의미·단위·결측 표기, 행 하나가 무엇을 뜻하는지(측정 한 번인가, 시료 하나인가), 알려진 문제(장비 교체 시점, 보정 오류 구간), 분할 방법과 그 이유(6.6장의 누수), 라이선스와 사용 조건을 적는다. 머신러닝 커뮤니티에서는 데이터셋마다 이런 내용을 정해진 질문 목록으로 정리하는 "데이터셋 datasheet" 관행이 제안되어 있다.

| 변수 | 의미 | 단위 | 범위 | 결측 표기 |
|---|---|---|---|---|
| `temp_k` | 시료 표면 온도 | K | 250–400 | 빈 칸 |
| `flux_w_m2` | 측정 열유속 | W/m² | −5000–5000 | `-9999` |
| `sample_id` | 시료 번호(같은 시료의 반복 측정은 같은 값) | — | S001–S120 | 없음 |

**모델 설명.** 모델의 입력과 출력(shape·단위), 구조의 요약, 학습 데이터의 범위(어떤 조건에서 학습했는가), 평가 방법과 결과, **쓰면 안 되는 범위**(학습 범위 밖의 온도, 학습에 없던 재료)를 적는다. 모델을 다른 사람이 쓰게 될 때 특히 중요하다. 같은 생각을 정해진 형식으로 정리한 "모델 카드(model card)" 관행이 있다.

**실험 기록.** 실험마다 질문과 가설, 실행한 run ID와 설정, 결과, 해석, 다음에 할 일을 남긴다. 실험 추적 도구(22.1장)가 숫자를 기록한다면, 실험 기록은 **왜 그 실험을 했고 무엇을 배웠는가**를 기록한다. 자세한 방법은 23.7장에서 다룬다.

**결정 기록**(ADR, Architecture Decision Record). 되돌리기 어렵거나 나중에 "왜?"라는 질문을 받을 결정(데이터 분할 단위, 평가 metric, 핵심 라이브러리, 정규화 방식)마다 짧은 문서 하나를 쓴다. 형식은 다음과 같다.

```markdown
# 0003. 데이터를 시료 단위로 분할한다
- 상태: 채택
- 날짜: 2026-10-12

## 맥락
시료 120개를 각각 5–20회 측정했다. 행 단위로 나누면 같은 시료가 train과 test에 함께 들어간다.

## 결정
sample_id 단위로 train/val/test = 80/20/20 시료로 나눈다. 시드는 configs/split.yaml에 고정한다.

## 검토한 대안
- 행 단위 무작위 분할: 데이터를 더 쓰지만 test 점수가 새 시료 성능을 과대평가한다(실험 E-07에서 MAE 차이 약 2배).
- 측정 날짜 단위 분할: 장비 상태 변화를 반영하지만 날짜별 시료 수가 고르지 않다.

## 결과
- 보고하는 test 점수는 "처음 보는 시료"에 대한 성능이다.
- train 데이터가 행 단위 분할보다 적다.
```

ADR은 결정을 **바꾸지 않는** 문서다. 결정이 바뀌면 이전 ADR을 고치지 않고 새 ADR을 쓴 뒤, 이전 것의 상태를 "대체됨(superseded by 0007)"으로 바꾼다. 그래야 "예전에는 왜 다르게 했지?"에도 답할 수 있다. 이 책의 저장소도 같은 방식으로 `editorial/decisions/`에 번호 붙은 결정 기록(상태·날짜·맥락·결정·결과)을 둔다.

::quiz{id="adr-purpose"}

:::note[분야 예시]
신약 후보 분자의 활성을 예측하는 모델을 다른 팀에 넘길 때, 모델 설명의 "쓰면 안 되는 범위"가 가장 중요한 정보가 되기도 한다. 학습 데이터가 특정 화학 계열(scaffold)에 몰려 있으면, 그 밖의 분자에 대한 예측은 수치가 나오더라도 근거가 약하다. 학습 데이터의 분자량·원소 구성 범위, 학습에 쓴 분석법(assay)과 측정 단위(IC50인지 pIC50인지)를 적어 두면, 받는 쪽이 모델을 범위 밖에서 쓰거나 단위를 바꿔 읽는 실수를 막을 수 있다.
:::

## 변경 기록

git의 커밋 기록은 모든 변경을 담지만, 사람이 읽기에는 너무 촘촘하다. 오타 수정과 결과를 바꾸는 수정이 같은 무게로 섞여 있다. **CHANGELOG**는 사용자(동료, 미래의 자신, 결과를 재현하려는 사람)에게 의미 있는 변경만 버전별로 골라 정리한 파일이다. 널리 쓰이는 관례인 Keep a Changelog는 버전마다 `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security` 항목으로 나누고, 아직 배포하지 않은 변경은 `Unreleased` 절에 모은다. 이 책의 저장소도 이 형식의 `CHANGELOG.md`를 둔다. 버전 번호는 흔히 Semantic Versioning(MAJOR.MINOR.PATCH)을 따른다. 호환되지 않는 변경이면 MAJOR, 기능 추가면 MINOR, 버그 수정이면 PATCH를 올린다.

연구 코드의 변경 기록에서 가장 중요한 것은 **결과에 영향을 준 변경을 따로 표시하는 것**이다. 소프트웨어의 기준으로는 작은 수정(상수 하나, 버그 하나)이 연구 결과를 바꿀 수 있다. 앞의 열전도도 기본값 변경은 모든 열유속을 약 3% 줄인다. metric 집계 버그를 고치면(6.6장) 이전 표의 숫자가 모두 바뀐다.

```markdown
## [0.4.0] - 2026-11-02

### Changed
- **[결과 영향]** 기본 열전도도를 0.6에서 0.58 W/(m K)로 고쳤다(측정값 반영, ADR 0005).
  모든 열유속 출력이 약 3% 줄어든다. v0.3.x로 만든 그림 3, 표 2를 다시 계산해야 한다.

### Fixed
- **[결과 영향]** 평가 MAE를 batch 평균의 평균으로 계산하던 버그를 고쳤다(#42).
  마지막 batch가 작을 때 MAE가 달라졌다. 영향받는 run: 2026-10-01 이후 batch_size=32인 평가 전부.
- README의 학습 명령에서 옛 인자 `--lr`을 `--learning-rate`로 고쳤다.
```

결과 영향 항목에는 세 가지를 쓴다. 무엇이 바뀌었는가, 어떤 결과가 얼마나 영향을 받는가, 어느 버전·커밋부터인가. 그리고 결과 파일에는 그 결과를 만든 코드 버전(커밋 hash)이 남아 있어야 한다(6.3장, 6.5장). 그래야 "이 그림은 버그 수정 전에 만든 것인가"에 답할 수 있다. 논문 제출본처럼 고정해야 하는 시점에는 태그(`v1.0`)를 붙이고 README에 적는다.

::quiz{id="changelog-result-impact"}

## 실습

자신의 프로젝트 README를, 처음 보는 사람이 그대로 따라 해 결과를 재현할 수 있게 만든다. 문서를 쓴 사람이 확인하는 것으로는 부족하다. 작성자의 컴퓨터에는 문서에 적지 않은 것(이미 받은 데이터, 셸 설정, 전에 설치한 패키지)이 남아 있기 때문이다.

1. 위 뼈대를 참고해 README의 여섯 절(목적·설치·데이터 준비·실행·결과 재현·인용)을 채운다. 결과 재현 절에는 대표 결과 하나 이상의 **기대 결과와 실행 시간**을 적는다.
2. 핵심 함수 다섯 개 이상에 NumPy 스타일 docstring을 쓴다. 인자와 반환값의 shape·단위, 예외, 실행 가능한 예제를 포함한다. `pytest --doctest-modules`로 예제를 실행한다.
3. 데이터 설명서(`docs/data.md`)와 결정 기록 하나(`docs/decisions/0001-....md`)를 쓴다. 결정 기록의 주제는 데이터 분할이나 평가 metric처럼 결과에 영향을 주는 결정으로 고른다.
4. README의 명령을 검사하는 방법을 하나 정해 CI에 넣는다(6.6장). 가장 좋은 것은 작은 설정으로 실제 실행하는 것이고, 최소한 앞의 코드처럼 CLI 인자를 검사한다.
5. **새 환경 시험**: 프로젝트를 처음 보는 동료에게 README 링크만 주고, 새 머신(또는 새 컨테이너, 새 계정)에서 설치부터 대표 결과 재현까지 해 보게 한다. 작성자는 옆에서 말하지 않고, 동료가 막힌 곳과 추측한 곳을 모두 적는다. 적힌 곳마다 README를 고치고, 막힘 없이 끝날 때까지 반복한다.

이어서 연습문제를 푼다. docstring의 `Parameters` 절과 함수의 실제 signature를 비교해, 문서에 빠진 인자와 문서에만 남은 낡은 인자를 찾는 검사기를 작성한다. 이 검사기를 테스트나 CI에 넣으면 인자 이름을 바꾸고 문서를 고치지 않은 변경이 바로 드러난다.

::exercise{id="docstring-drift"}

## 실패 사례

- **실행되지 않는 예제 명령.** README의 명령이 몇 달 전 인자 이름을 쓰거나, 지금은 없는 스크립트를 가리키거나, 작성자 컴퓨터에만 있는 경로(`/home/kim/data`)를 쓴다. 독자는 첫 단계에서 막히고 README 전체를 믿지 않게 된다. 명령을 CI에서 실행하거나 검사하고, 경로는 프로젝트 루트 기준 상대 경로나 설정 값으로 쓴다.
- **코드와 어긋난 문서.** docstring은 두께를 m라고 하는데 코드는 mm를 받고, README의 기대 결과는 버그 수정 전의 숫자다. 틀린 문서는 문서가 없는 것보다 나쁘다. 독자가 믿고 따르기 때문이다. 실행 가능한 예제(doctest), signature와 docstring 비교, README 명령 검사로 어긋남을 기계적으로 잡고, 코드를 바꾸는 PR에서 문서를 함께 고친다.
- **기대 결과가 없는 재현 절차.** 명령만 있고 무엇이 나와야 하는지 없으면, 따라 한 사람은 성공했는지 판단할 수 없다. 숫자와 허용 범위, 실행 시간을 적는다.
- **"왜"가 남지 않은 결정.** 데이터 분할 단위, 제외한 시료, 평가 metric의 선택 이유가 작성자의 기억에만 있으면, 후임자는 그 결정을 모르고 바꾸거나, 이유를 모른 채 따르기만 한다. 결과에 영향을 주는 결정은 ADR로 남긴다.
- **결과를 바꾼 변경을 기록하지 않는 것.** 상수 하나, 버그 하나를 고친 뒤 이전 결과와 새 결과가 섞여 표 하나에 들어간다. CHANGELOG에 결과 영향을 표시하고, 결과 파일에 코드 버전을 남긴다.

## 통과 기준

- 새 연구원이 구두 설명 없이 README만으로 첫 결과를 재현한다. 재현한 숫자가 README의 기대 결과 범위 안에 있다.
- 핵심 함수의 docstring에 shape·단위·예외와 실행 가능한 예제가 있고, `pytest --doctest-modules`가 통과한다.
- 결과에 영향을 주는 결정 하나 이상을 결정 기록으로 남기고, 결과에 영향을 준 변경을 CHANGELOG에 표시한다.
- 연습문제 `docstring-drift`의 테스트를 통과한다.

## 참고 문헌

- Python 공식 문서, *doctest — Test interactive Python examples* (docs.python.org/3/library/doctest.html). — 예제 실행과 출력 비교 방식
- PEP 257 — *Docstring Conventions* (peps.python.org/pep-0257). — docstring의 기본 관례
- numpydoc 문서, *Style guide* (numpydoc.readthedocs.io). — NumPy 스타일 docstring의 절 구성
