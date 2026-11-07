# 데이터셋 manifest와 해시 만들기

데이터셋의 버전을 이름(`v2_final_real.csv`)이 아니라 **내용**으로 고정한다. 파일마다 내용 해시(sha256)를 구해 목록(manifest)으로 만들고, 그 목록을 다시 해시하면 데이터셋 전체를 대표하는 값 하나가 나온다. 파일 하나의 1 byte만 바뀌어도, 파일 이름 하나만 바뀌어도 이 값이 바뀐다.

함수 `dataset_manifest(files)`를 완성한다. `files`는 `{경로: 내용 bytes}` 딕셔너리다. `(manifest, digest)`를 반환한다.

**manifest 형식.** 파일마다 한 줄, 경로의 문자열 순서로 정렬한다. 한 줄은

```
<내용의 sha256 hex>  <byte 수>  <경로>\n
```

이다. 구분자는 공백 **두 칸**이다(`sha256sum` 출력과 비슷한 형식). 줄마다 `\n`으로 끝난다.

**digest.** manifest 문자열 전체를 UTF-8로 인코딩한 바이트의 sha256 hex다.

예: `{"empty.txt": b""}`의 manifest는

```
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  0  empty.txt
```

다(빈 바이트열의 sha256).

정렬이 왜 필요한지 생각해 본다. 딕셔너리나 디렉터리 목록의 순서는 도구와 운영체제에 따라 다를 수 있다. 같은 내용이 같은 해시를 내려면 순서를 정해 두어야 한다.

힌트: `hashlib.sha256(data).hexdigest()`, `sorted(files)`.
