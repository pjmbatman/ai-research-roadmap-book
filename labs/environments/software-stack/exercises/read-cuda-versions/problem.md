# 세 가지 CUDA 버전 읽고 판정하기

한 서버에서 다음 세 출력을 얻었다.

- `nvidia-smi` 출력 (드라이버가 없으면 `None`)
- `nvcc --version` 출력 (CUDA toolkit이 없으면 `None`)
- `torch.version.cuda` 값 (CPU 전용 빌드이면 `None`)

함수 `summarize(smi_text, nvcc_text, torch_cuda)`를 완성해 다음 키를 가진 dict를 반환한다.

| 키 | 값 |
|---|---|
| `driver` | 드라이버 버전 문자열(예: `"550.54.15"`) 또는 `None` |
| `driver_cuda` | 드라이버가 지원하는 최대 CUDA 버전(예: `"12.4"`) 또는 `None` |
| `toolkit` | nvcc가 보고하는 toolkit 버전(예: `"12.1"`) 또는 `None` |
| `torch_ok` | 아래 규칙의 판정 결과 |

`torch_ok` 규칙(보수적 판정):

- `torch_cuda`가 `None`(CPU 빌드)이면 `None`.
- 드라이버가 없으면 `False`.
- 그 밖에는 `torch_cuda`의 (major, minor)가 `driver_cuda`의 (major, minor) **이하**이면 `True`, 아니면 `False`.

toolkit은 판정에 쓰지 않는다. PyTorch wheel은 CUDA runtime을 함께 가져오기 때문이다.

힌트: 버전은 문자열로 비교하지 않는다. 문자열 비교에서는 `"12.10" < "12.4"`가 `True`다. 정수 tuple로 바꿔 비교한다.
