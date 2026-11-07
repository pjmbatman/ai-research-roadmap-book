# 근거 규칙으로 다음 성장 목표 고르기

역량 루브릭을 채우고 나면 "다음에 무엇을 할까"를 정해야 한다. 이 장의 원칙은 두 가지다. 첫째, **근거로 보일 수 있는 수준까지만 인정한다.** 스스로 4라고 매겼어도 근거가 1까지만 있으면 그 항목은 1이다. 둘째, **자기 연구에 필요하면서 가장 낮은 칸부터** 채운다.

함수 `next_targets(rows, k=2)`를 완성한다. `rows`는 루브릭의 각 항목을 담은 dict의 list다.

| 키 | 뜻 |
|---|---|
| `"item"` | 항목 이름 (str) |
| `"claimed"` | 스스로 매긴 수준 (0–5의 int) |
| `"evidenced"` | 근거 파일로 보일 수 있는 최고 수준 (0–5의 int) |
| `"needed"` | 자기 연구에 필요한 항목인가 (bool) |

반환값은 항목 이름의 list다.

- 각 항목의 수준은 $\min(\text{claimed}, \text{evidenced})$로 본다.
- `needed`가 `True`인 항목만 고른다.
- 수준이 낮은 순서로 최대 `k`개를 반환한다. 수준이 같으면 `rows`에 나온 순서를 따른다.
- `claimed`나 `evidenced`가 0–5의 정수가 아니면 `ValueError`를 낸다(`2.5`, `6`, `-1` 등).

예:

```python
rows = [
    {"item": "실패 원인 분리", "claimed": 4, "evidenced": 1, "needed": True},
    {"item": "주장 판단",     "claimed": 2, "evidenced": 2, "needed": True},
    {"item": "논문 연결",     "claimed": 3, "evidenced": 3, "needed": True},
]
next_targets(rows)   # ["실패 원인 분리", "주장 판단"]
```

힌트: `(수준, 원래 위치, 이름)`의 tuple을 만들어 정렬하면 "수준이 같으면 원래 순서" 규칙이 자동으로 지켜진다. Python의 `sorted`는 안정 정렬이므로 수준만 key로 써도 된다.
