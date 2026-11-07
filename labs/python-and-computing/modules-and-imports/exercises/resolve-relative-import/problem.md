# 상대 import를 절대 이름으로 바꾸기

`from ..utils import load`처럼 점으로 시작하는 상대 import는, import하는 모듈이 속한 **패키지 이름**(`__package__`)을 기준으로 절대 이름으로 바뀐 뒤 찾아진다. 이 변환을 함수 `resolve(name, package)`로 직접 구현한다.

| `name` | `package` (`__package__`) | 결과 |
|---|---|---|
| `".physics"` | `"heatlab.models"` | `"heatlab.models.physics"` |
| `"..utils"` | `"heatlab.models"` | `"heatlab.utils"` |
| `"..io.hdf5"` | `"heatlab.models"` | `"heatlab.io.hdf5"` |
| `"."` | `"heatlab.models"` | `"heatlab.models"` |
| `"numpy"` | `"heatlab.models"` | `"numpy"` (점이 없으면 절대 import) |

규칙은 다음과 같다.

- 앞의 점 개수가 level이다. 점 하나는 현재 패키지, 점 둘은 한 단계 위 패키지다. 즉 `package`의 끝에서 `level - 1`개 구성 요소를 떼어 낸 것이 기준 패키지다.
- 점 뒤에 이름이 있으면 기준 패키지 뒤에 `.`으로 잇고, 없으면 기준 패키지 자체가 결과다.
- 상대 이름인데 `package`가 `None`이거나 빈 문자열이면 `ImportError`를 낸다. `python pkg/mod.py`처럼 파일 경로로 실행한 모듈이 상대 import에서 실패하는 이유다(그 모듈의 `__package__`가 비어 있다).
- 떼어 낼 구성 요소가 모자라면(최상위 패키지보다 위로 올라가면) `ImportError`를 낸다.

`importlib.util.resolve_name`이 같은 일을 하지만, 여기서는 쓰지 않고 문자열 처리로 구현한다. 테스트가 여러 경우에서 결과를 `importlib.util.resolve_name`과 비교한다.

주의: 두 번째 인자는 모듈 이름이 아니라 **패키지** 이름이다. 모듈 `heatlab/models/cnn.py`의 `__package__`는 `"heatlab.models"`다.
