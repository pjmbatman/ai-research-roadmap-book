# 설정에서 run ID 만들기

실험 결과는 `outputs/<run-id>/` 아래에 모은다. run ID가 **언제** 시작했는지와 **어떤 설정**으로 실행했는지를 함께 담으면, 디렉터리 이름만 보고도 같은 설정의 실행을 묶을 수 있다.

함수 `make_run_id(config, started, ignore=())`를 완성한다.

- 반환 형식은 `"YYYYmmdd-HHMMSS-xxxxxxxx"`다. 앞부분은 `started`(`datetime`)의 날짜·시각, 뒤의 8자리는 설정 hash다.
- 설정 hash는 다음 순서로 만든다.
  1. `ignore`에 적힌 점 경로(dotted key, 예: `"paths.output_dir"`, `"logging"`)를 설정에서 뺀다. 결과에 영향을 주지 않는 값(출력 위치, 로그 수준 등)이 hash를 바꾸지 않게 하기 위해서다. 설정에 없는 경로는 무시한다.
  2. `json.dumps(cfg, sort_keys=True, separators=(",", ":"))`로 직렬화한다. 키 순서가 달라도 같은 문자열이 나온다.
  3. UTF-8로 인코딩한 문자열의 SHA-256 16진 digest 앞 8자리를 쓴다.
- 인자로 받은 `config`를 바꾸지 않는다. 호출한 쪽의 설정이 몰래 바뀌면, 그 설정을 저장하는 다음 단계가 틀린 값을 기록한다.

```python
from datetime import datetime
make_run_id({"train": {"lr": 0.001, "seed": 0}}, datetime(2026, 11, 7, 14, 3, 9))
# '20261107-140309-' 뒤에 8자리 hash
```

힌트: 중첩 dict를 바꾸지 않고 키를 지우려면 먼저 `copy.deepcopy`로 복사한다.
