# PATH에서 실행될 파일 찾기

여러 CUDA toolkit이 공존하는 서버에서 `nvcc`를 입력하면 어느 파일이 실행될까. 셸은 `PATH`의 디렉터리를 **앞에서부터** 차례로 보고 처음 찾은 파일을 실행한다. 동적 링커가 `LD_LIBRARY_PATH`에서 라이브러리를 찾을 때도 같은 규칙(앞에서부터, 처음 찾은 것)을 쓴다.

함수 `lookup(name, path_var, files, links)`를 완성한다.

| 인자 | 의미 |
|---|---|
| `name` | 찾을 파일 이름(예: `"nvcc"`, `"libcudart.so.12"`) |
| `path_var` | 콜론(`:`)으로 구분한 디렉터리 목록 문자열 |
| `files` | 실제로 존재하는 파일의 절대 경로 집합 |
| `links` | 디렉터리 심볼릭 링크 dict, `{링크 경로: 실제 경로}` |

반환값은 `(찾은 경로, 실제 경로)` tuple이다.

- 찾은 경로: `PATH` 항목 그대로 이어 붙인 경로(예: `"/usr/local/cuda/bin/nvcc"`)
- 실제 경로: 링크를 따라간 경로(예: `"/usr/local/cuda-12.4/bin/nvcc"`)
- 어디에도 없으면 `None`

규칙:

- 빈 항목(`"a::b"`의 가운데)은 건너뛴다.
- 항목 끝의 `/`는 무시한다(`"/usr/bin/"`과 `"/usr/bin"`은 같다).
- 링크는 경로의 앞부분이 링크 경로와 디렉터리 단위로 일치할 때만 바꾼다. `/usr/local/cuda` 링크는 `/usr/local/cuda/bin`에는 적용되지만 `/usr/local/cuda-12.1/bin`에는 적용되지 않는다.
