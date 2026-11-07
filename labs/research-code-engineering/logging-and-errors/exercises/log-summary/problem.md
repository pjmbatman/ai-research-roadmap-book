# JSON 줄 로그 요약하기

학습이 밤사이 멈췄다. 남은 것은 run 디렉터리의 `metrics.jsonl`뿐이다. 한 줄이 JSON 객체 하나인 로그(JSON Lines)를 읽어, 실패 원인을 찾는 데 필요한 요약을 만드는 함수 `summarize(lines)`를 완성한다.

입력 `lines`는 문자열 리스트다(파일의 각 줄). 예:

```text
{"level": "INFO", "event": "config", "lr": 0.3, "seed": 0}
{"level": "INFO", "event": "train", "step": 0, "loss": 2.31}
{"level": "WARNING", "event": "grad_clip", "step": 1, "norm": 54.2}
{"level": "INFO", "event": "train", "step": 2, "loss": NaN}
{"level": "INFO", "event": "tr
```

반환값은 다음 키를 가진 dict다.

| 키 | 값 |
|---|---|
| `"levels"` | 레벨 이름 → 그 레벨의 레코드 수. `"level"` 키가 있는 레코드만 센다 |
| `"first_bad_step"` | 로그 순서상 **처음으로** `"loss"`가 유한하지 않은 레코드의 `"step"`. 없으면 `None` |
| `"config"` | `"event"`가 `"config"`인 첫 레코드에서 `"level"`과 `"event"`를 뺀 dict. 없으면 `None` |
| `"malformed"` | JSON 객체로 읽을 수 없는 줄의 수 |

규칙:

- 빈 줄(공백뿐인 줄 포함)은 건너뛴다. `malformed`에 세지 않는다.
- JSON으로 읽을 수 없는 줄(강제 종료로 잘린 마지막 줄, 누군가 `print`로 섞어 넣은 줄)과, JSON이지만 객체(dict)가 아닌 줄은 `malformed`로 센다. **예외로 멈추지 않는다.** 망가진 로그를 읽는 도구가 망가지면 안 된다.
- `"loss"`가 `NaN`, `Infinity`, `-Infinity`이거나 `null`이면 유한하지 않은 것으로 본다. Python의 `json.loads`는 `NaN`과 `Infinity`를 float로 읽는다.
- `"loss"` 키가 없는 레코드는 `first_bad_step` 판단에서 제외한다.

힌트: `json.JSONDecodeError`는 `ValueError`의 하위 클래스다. 잡을 예외를 좁게 고른다. 모든 예외를 잡으면 이 함수 자신의 버그까지 malformed로 숨겨진다.
