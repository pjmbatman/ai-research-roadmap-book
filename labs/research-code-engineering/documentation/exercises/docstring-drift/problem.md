# docstring과 코드의 어긋남 찾기

인자 이름을 바꾸거나 인자를 추가하면서 docstring을 고치지 않는 일은 흔하다. 문서는 오류를 내지 않으므로 아무도 모른 채 몇 달이 지난다. 함수 `doc_drift(func)`를 완성해, NumPy 스타일 docstring의 `Parameters` 절과 함수의 실제 signature를 비교한다.

반환값: `{"missing": [...], "stale": [...]}` — 두 리스트 모두 이름의 오름차순 정렬.

- `missing`: signature에는 있는데 문서에 없는 인자
- `stale`: 문서에는 있는데 signature에 없는 인자(이름이 바뀌었거나 지워진 인자)

규칙:

1. signature의 인자는 `inspect.signature(func).parameters`에서 얻는다. `self`와 `cls`는 제외한다. `*args`, `**kwargs`는 별표를 뗀 이름(`args`, `kwargs`)으로 비교한다.
2. docstring은 `inspect.getdoc(func)`로 얻는다(들여쓰기가 정리된 문자열). docstring이 없거나 `Parameters` 절이 없으면 문서화된 인자는 없는 것으로 본다.
3. 절(section)은 제목 줄 다음 줄이 `-`로만 이루어진 줄(3자 이상)인 것이다. 예:

   ```text
   Parameters
   ----------
   x, y : ndarray, shape (N,)
       좌표 [m].
   *args : float
       추가 계수.
   ```

4. `Parameters` 절은 그 밑줄 다음 줄부터 **다음 절의 제목 줄 직전** 또는 docstring 끝까지다.
5. 절 안에서 **들여쓰기 없이 시작하는 비어 있지 않은 줄**이 인자 줄이다. 들여쓴 줄은 설명이므로 그 안에 `:`가 있어도 무시한다.
6. 인자 줄의 이름 부분은 첫 `:` 앞이다(`:`가 없으면 줄 전체). 쉼표로 여러 이름을 적을 수 있다. 각 이름의 앞뒤 공백과 앞의 `*`를 뗀다.

힌트: 다음 절의 제목 줄인지 판단하려면 그 다음 줄을 함께 봐야 한다.
