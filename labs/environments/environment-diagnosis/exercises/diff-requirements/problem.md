# 업그레이드 전후 패키지 목록 비교하기

"어제까지 되던 실험이 오늘 안 된다." 변경 관리를 했다면 업그레이드 전후에 `pip freeze`(또는 `uv pip freeze`) 결과를 남겨 두었을 것이다. 두 목록을 비교하면 무엇이 바뀌었는지가 후보로 좁혀진다. 함수 `diff_requirements(before, after)`를 완성한다.

입력은 freeze 형식의 **문자열** 두 개다. 한 줄에 하나씩 다음 형식이 온다.

```text
# 주석 줄
numpy==2.1.3
Pillow==11.0.0
mypkg @ file:///home/me/mypkg
-e git+https://example.org/repo.git#egg=tool
```

규칙:

1. 빈 줄, `#`로 시작하는 주석 줄, `-`로 시작하는 옵션 줄(`-e`, `--index-url` 등)은 무시한다.
2. `이름==버전` 줄은 이름과 버전으로 나눈다. `이름 @ 주소` 줄은 `@` 뒤의 주소 문자열을 버전 자리에 쓴다.
3. 패키지 이름은 비교 전에 정규화한다: 소문자로 바꾸고, `_`와 `.`을 `-`로 바꾼다(`Pillow`와 `pillow`, `ruamel.yaml`과 `ruamel-yaml`은 같은 패키지).
4. 줄 앞뒤의 공백은 무시한다.

반환값은 정규화한 이름을 키로 쓰는 dict 세 개를 담은 dict다.

```python
{
    "added":   {이름: 새 버전},            # after에만 있다
    "removed": {이름: 옛 버전},            # before에만 있다
    "changed": {이름: (옛 버전, 새 버전)},  # 둘 다 있고 버전이 다르다
}
```

버전이 같은 패키지는 어디에도 넣지 않는다.
