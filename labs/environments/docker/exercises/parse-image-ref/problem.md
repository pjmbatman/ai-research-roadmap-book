# image 참조 해석하기

`docker pull ubuntu`의 `ubuntu`는 줄임말이다. Docker는 이것을 `docker.io/library/ubuntu:latest`로 해석한다. 연구 기록에 image를 적을 때는 이렇게 **완전한 참조**로 적어야 다른 사람이 같은 image를 찾을 수 있다. 함수 `parse(ref)`를 완성해 image 참조를 네 부분으로 나눈다.

반환값: `{"registry": ..., "repository": ..., "tag": ..., "digest": ...}`

규칙(Docker의 참조 해석 규칙을 단순화한 것):

1. `@`가 있으면 그 뒤가 digest(예: `sha256:...`)다. 없으면 `digest`는 `None`.
2. 남은 부분에서 **마지막 `/` 뒤에 있는** `:` 다음이 tag다. `localhost:5000/proj`의 `:5000`은 tag가 아니라 registry의 포트다.
3. tag도 digest도 없으면 tag는 `"latest"`다. digest만 있으면 tag는 `None`이다.
4. `/`로 나눈 첫 부분에 `.`이나 `:`가 있거나 그것이 `localhost`이면 registry다. 아니면 registry는 `"docker.io"`다.
5. registry가 `docker.io`이고 repository에 `/`가 없으면 앞에 `library/`를 붙인다(공식 image).

| 입력 | registry | repository | tag | digest |
|---|---|---|---|---|
| `ubuntu` | `docker.io` | `library/ubuntu` | `latest` | `None` |
| `python:3.12-slim` | `docker.io` | `library/python` | `3.12-slim` | `None` |
| `ghcr.io/astral-sh/uv:latest` | `ghcr.io` | `astral-sh/uv` | `latest` | `None` |
| `localhost:5000/proj` | `localhost:5000` | `proj` | `latest` | `None` |

힌트: digest를 먼저 떼고, 그다음 tag, 마지막으로 registry를 정한다.
