---
id: environments/docker
title: "Docker와 컨테이너"
title_en: "Docker and Containers"
tier: core
status: draft
prerequisites: [environments/dependencies-and-packaging]
---

# Docker와 컨테이너

3.4장의 lockfile은 Python 패키지를 고정한다. 그러나 환경은 Python 패키지만으로 이루어지지 않는다. 시스템 라이브러리(glibc, apt로 설치한 C 라이브러리), Python 인터프리터 자체, 환경변수, 실행 명령까지 모두 결과에 영향을 줄 수 있다. **컨테이너**는 이것들을 OS 수준에서 한 덩어리로 묶어, 다른 머신에서도 같은 조건으로 실행하게 한다. "이 image로 이 명령을 실행하면 된다"는 한 줄이 설치 문서 몇 쪽을 대신한다.

이 장은 Docker를 기준으로 image와 container의 차이, 연구용 Dockerfile 작성, 데이터와 결과를 컨테이너 밖에 두는 원칙, GPU 사용, 공유 서버·클러스터에서의 실무 문제를 다룬다. 명령과 image tag는 집필 시점의 예시이며, 이 장의 Docker 명령은 집필 환경에 Docker daemon이 없어 직접 실행해 확인하지 못했다. 실행 전에 Docker와 NVIDIA Container Toolkit의 공식 문서로 확인한다.

## 학습 목표

- image와 container를 구분한다
- GPU를 사용하는 연구용 Dockerfile을 작성한다
- 데이터와 결과를 컨테이너 밖에서 관리한다

## 기본 개념

컨테이너는 가상머신(VM)과 다르다. VM은 하드웨어를 흉내 내고 그 위에 별도의 OS kernel을 띄운다. 컨테이너는 **호스트의 kernel을 공유**하면서, 파일시스템·프로세스·네트워크를 격리한 프로세스다. 그래서 가볍고 빨리 뜨지만, kernel과 kernel 모듈(GPU 드라이버 포함)은 호스트의 것을 쓴다. 이 사실이 뒤의 GPU 절의 출발점이다.

Docker의 다섯 가지 개념은 다음과 같다.

| 개념 | 정의 | 비유 |
|---|---|---|
| **image** | 읽기 전용 layer들을 쌓은 파일시스템과 실행 설정(기본 명령, 환경변수 등)의 묶음 | 설치 디스크 |
| **container** | image 위에 쓰기 가능한 layer를 하나 얹어 실행한 인스턴스 | 설치 디스크로 띄운 실행 중인 시스템 |
| **registry** | image를 저장·배포하는 서버(Docker Hub, GitHub Container Registry, 기관 내부 registry 등) | 저장소 |
| **tag** | image를 가리키는 사람이 읽는 이름(`python:3.12-slim`, `lab/proj:v1`). 다른 image로 옮길 수 있다 | 브랜치 이름 |
| **digest** | image 내용의 SHA-256 해시(`sha256:...`). 내용이 같으면 같고, 한 바이트라도 다르면 다르다 | 커밋 해시 |

image 하나로 컨테이너를 여러 개 띄울 수 있다. 컨테이너 안에서 파일을 고치면 그 컨테이너의 쓰기 layer에만 기록되고, image와 다른 컨테이너는 그대로다. 컨테이너를 지우면 쓰기 layer도 사라진다. 이 성질이 "실험이 환경을 더럽히지 않는다"는 장점이자, "결과를 컨테이너 안에 저장하면 사라진다"는 함정이다.

::quiz{id="image-vs-container"}

tag와 digest의 차이는 재현에서 결정적이다. tag는 registry에서 언제든 다른 image로 옮길 수 있는 이름이고, digest는 내용 자체에서 계산된다. 이 관계를 해시로 흉내 낸다(실제 image digest는 image의 manifest 파일 내용의 SHA-256이다).

```python {exec="browser" session="docker"}
import hashlib

def digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()[:12] + "..."

v1 = b'{"layers": ["ubuntu-base", "python-3.12", "numpy-2.5.4"]}'
v2 = b'{"layers": ["ubuntu-base", "python-3.12", "numpy-2.5.5"]}'   # 한 글자 차이

registry = {"lab/proj:latest": digest(v1)}
print("1월 push  lab/proj:latest ->", registry["lab/proj:latest"])
registry["lab/proj:latest"] = digest(v2)                             # 같은 tag로 다시 push
print("6월 push  lab/proj:latest ->", registry["lab/proj:latest"])
print("1월 내용을 digest로 다시 계산 ->", digest(v1))
```

```output
1월 push  lab/proj:latest -> sha256:4a2eddfbe00e...
6월 push  lab/proj:latest -> sha256:c20fdab807d9...
1월 내용을 digest로 다시 계산 -> sha256:4a2eddfbe00e...
```

같은 이름 `lab/proj:latest`가 6월에는 다른 내용을 가리킨다. 반면 1월 내용의 digest는 언제 계산해도 같다. 그래서 실험 기록에는 tag가 아니라 `lab/proj@sha256:...` 형식의 digest를 남긴다. 로컬 image의 digest는 `docker image inspect --format '{{json .RepoDigests}}' lab/proj:latest`로 확인한다(registry에 push했거나 registry에서 pull한 image에만 있다).

## Dockerfile

**Dockerfile**은 image를 만드는 절차를 적은 파일이다. 각 줄(instruction)이 실행되어 layer 하나를 만든다. 연구용 Python 프로젝트의 Dockerfile은 대체로 다음 구조를 따른다(집필 시점 예시. 이 책의 실습 파일 `labs/environments/docker/Dockerfile`과 같은 구조다).

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim

# uv 실행 파일만 공식 이미지에서 복사한다
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# 1) 의존성 기록만 먼저 복사해 설치한다
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2) 코드는 마지막에 복사한다
COPY . .
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "train.py"]
```

**base image 선택.** `FROM`이 정하는 base image가 OS·glibc·시스템 라이브러리를 정한다. GPU를 쓰는 연구용 image의 base는 두 계열로 나뉜다.

- **일반 OS·Python image**(예: `python:3.12-slim`, `ubuntu:24.04`): PyTorch를 pip·uv wheel로 설치하면 CUDA runtime이 wheel과 함께 들어오므로(3.1장, 3.4장), base에 CUDA가 없어도 GPU 학습이 된다. image가 작고 단순하다.
- **CUDA image 계열**(NVIDIA가 배포하는 `nvidia/cuda` image 등): CUDA 라이브러리가 미리 들어 있다. 보통 `base`(최소), `runtime`(실행 라이브러리), `devel`(컴파일러 `nvcc`와 헤더 포함) 같은 변형과 cuDNN 포함 여부, Ubuntu 버전이 tag에 표시된다. custom CUDA extension을 빌드해야 하면 `devel` 변형이 필요하다. 어떤 tag가 있는지는 registry 페이지에서 확인한다.

**layer와 build cache.** Docker는 각 줄의 결과를 cache하고, 다음 빌드에서 그 줄의 입력(명령 문자열, 복사하는 파일의 내용)이 같으면 다시 실행하지 않는다. 어떤 줄의 입력이 바뀌면 **그 줄과 그 뒤의 모든 줄**을 다시 실행한다. 그래서 줄의 순서가 빌드 시간을 결정한다. 자주 바뀌는 것(코드)을 뒤에, 드물게 바뀌는 것(의존성 기록)을 앞에 둔다. 두 순서를 비교한다.

```python {exec="browser" session="docker"}
def rebuilt(dockerfile, changed):
    """changed 파일이 바뀌었을 때 다시 실행되는 줄. 한 줄이 무효화되면 그 뒤는 모두 다시 실행된다."""
    for i, (line, inputs) in enumerate(dockerfile):
        if inputs & changed:
            return [l for l, _ in dockerfile[i:]]
    return []

code_first = [("FROM python:3.12-slim", set()),
              ("COPY . /app", {"pyproject.toml", "uv.lock", "train.py"}),
              ("RUN uv sync --locked", set())]
lock_first = [("FROM python:3.12-slim", set()),
              ("COPY pyproject.toml uv.lock /app/", {"pyproject.toml", "uv.lock"}),
              ("RUN uv sync --locked --no-install-project", set()),
              ("COPY . /app", {"pyproject.toml", "uv.lock", "train.py"})]

for name, df in [("코드 먼저", code_first), ("lockfile 먼저", lock_first)]:
    print(f"[{name}] train.py 수정 -> 다시 실행:", rebuilt(df, {"train.py"}))
```

```output
[코드 먼저] train.py 수정 -> 다시 실행: ['COPY . /app', 'RUN uv sync --locked']
[lockfile 먼저] train.py 수정 -> 다시 실행: ['COPY . /app']
```

코드를 먼저 복사하면 `train.py` 한 줄을 고칠 때마다 의존성 설치(수 분, PyTorch라면 수 GB 다운로드)가 다시 실행된다. lockfile을 먼저 복사하면 의존성 layer는 cache에서 재사용되고 코드 복사만 다시 일어난다. (이 흉내에서는 생략했지만, 실제 Dockerfile에서는 코드를 복사한 뒤 프로젝트 자신을 설치하는 짧은 `uv sync` 줄이 뒤따른다.) `RUN --mount=type=cache`는 BuildKit 기능으로, 패키지 다운로드 cache를 빌드 사이에 보존하되 image에는 넣지 않는다.

::widget{name="docker-layer-cache"}

::quiz{id="layer-order"}

**`.dockerignore`.** `COPY . .`은 빌드 context(보통 저장소 루트) 전체를 복사한다. `.venv`, `.git`, 데이터, 결과 디렉터리가 섞여 들어가지 않도록 `.dockerignore`에 적어 둔다. 로컬의 `.venv`가 image에 복사되면 image가 커질 뿐 아니라, 다른 경로에서 만든 가상환경이라 제대로 동작하지도 않는다(3.3장).

**multi-stage build.** 빌드에만 필요한 도구(컴파일러, CUDA toolkit의 `devel` 변형)와 실행에 필요한 것을 단계로 나눈다. 첫 단계(`FROM ... AS builder`)에서 extension을 컴파일하고, 두 번째 단계에서 가벼운 runtime image에 결과물만 `COPY --from=builder`로 가져온다. 실행 image가 작아지고, 컴파일러 같은 불필요한 도구가 배포되지 않는다.

## 데이터와 결과

컨테이너의 원칙은 **image에는 환경만, 데이터와 결과는 밖에**다. 데이터를 image에 넣으면 image가 데이터 크기만큼 커지고, 데이터가 바뀔 때마다 다시 빌드해야 하며, image를 공유하는 순간 데이터도 함께 퍼진다. 결과를 컨테이너의 쓰기 layer에 저장하면 컨테이너와 함께 사라진다.

컨테이너 밖의 저장소를 연결하는 방법은 두 가지다.

| 방법 | 형식 | 용도 |
|---|---|---|
| **bind mount** | 호스트의 디렉터리를 컨테이너 경로에 연결 | 데이터 읽기, 코드 개발, 결과 저장. 호스트에서 바로 보인다 |
| **volume** | Docker가 관리하는 저장 공간 | 데이터베이스 파일, 컨테이너 사이에 공유하는 cache 등 |

연구에서는 bind mount가 기본이다. 결과가 호스트의 프로젝트 디렉터리에 바로 남고, 백업·버전 관리·다른 도구와의 연동이 쉽기 때문이다. 실행 예는 다음과 같다(집필 시점 예시).

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v /data/shared/dataset-v3:/data:ro \
    -v "$PWD/outputs":/outputs \
    -w /app \
    -e OUTPUT_DIR=/outputs \
    lab/proj@sha256:<digest> python train.py --data /data
```

- 데이터는 읽기 전용(`:ro`)으로 마운트한다. 학습 코드의 버그가 원본 데이터를 덮어쓰는 사고를 막는다.
- 결과 디렉터리는 쓰기 가능으로 마운트하고, 코드는 출력 경로를 환경변수나 인자로 받는다. 코드 안에 컨테이너 경로를 하드코딩하지 않으면 같은 코드가 컨테이너 밖에서도 돈다.
- `-w`(working directory)를 명시해, 상대 경로가 어디를 기준으로 하는지 분명히 한다.
- `--rm`은 끝난 컨테이너를 자동으로 지운다. 결과가 마운트한 경로에 있으므로 안심하고 지울 수 있다. 반대로 말하면, `--rm`을 쓰면서 마운트하지 않은 경로에 결과를 쓰면 모두 사라진다.

:::note[분야 예시]
의료 영상이나 유전체 데이터처럼 외부 반출이 제한된 데이터는 image에 절대 넣지 않는다. image는 다른 기관의 연구자에게 보낼 수 있지만 데이터는 보낼 수 없기 때문이다. 이런 공동 연구에서는 "image(환경과 코드)는 공유하고, 각 기관이 자기 데이터를 마운트해 같은 분석을 실행한다"는 방식이 쓰인다. 데이터를 밖에 두는 원칙이 협업 구조 자체를 가능하게 한다.
:::

## GPU 사용

컨테이너는 호스트 kernel을 공유하므로, GPU 드라이버의 kernel 모듈도 호스트의 것을 쓴다. 컨테이너 안에서 GPU를 쓰려면 호스트에 **NVIDIA Container Toolkit**을 설치하고 Docker가 그것을 쓰도록 설정한다. 그러면 `docker run --gpus ...`로 실행할 때 툴킷이 호스트 드라이버의 사용자 공간 라이브러리(`libcuda.so` 등)와 GPU 장치 파일을 컨테이너에 넣어 준다(설치 절차는 공식 설치 가이드를 따른다).

```bash
# 모든 GPU를 쓴다 / 특정 GPU만 쓴다 (집필 시점 예시)
docker run --rm --gpus all lab/proj nvidia-smi
docker run --rm --gpus '"device=0,1"' lab/proj python train.py
```

계층으로 보면 역할이 분명하다(3.1장).

| 층 | 어디에서 오는가 |
|---|---|
| GPU, kernel, 드라이버 kernel 모듈 | 호스트 |
| 드라이버의 사용자 공간 라이브러리(`libcuda.so`), `nvidia-smi` | 호스트(툴킷이 컨테이너에 주입) |
| CUDA runtime, cuDNN, NCCL | image(PyTorch wheel 또는 CUDA base image) |
| Python, PyTorch, 코드 | image |

따라서 규칙은 3.1장과 같다. **image 안의 CUDA runtime 버전이 호스트 드라이버가 지원하는 범위 안에 있어야 한다.** image에 드라이버를 설치하지 않는다. 호스트에 CUDA toolkit이 있을 필요도 없다. 호스트 드라이버가 CUDA 12.x까지만 지원하는데 CUDA 13 계열 PyTorch가 든 image를 쓰면, 컨테이너 안에서도 똑같이 실패한다. 컨테이너는 이 규칙을 없애지 않고, 규칙을 지켜야 하는 경계를 "호스트 드라이버 ↔ image"로 옮길 뿐이다.

::quiz{id="host-driver"}

## 실무 이슈

**사용자 권한(UID/GID).** 컨테이너 안의 프로세스는 기본적으로 root(UID 0)로 실행된다. 이 상태로 bind mount한 디렉터리에 결과를 쓰면, 호스트에서 그 파일의 소유자가 root가 되어 일반 사용자가 지우거나 고칠 수 없다. `--user "$(id -u):$(id -g)"`로 호스트 사용자와 같은 UID·GID로 실행하면 이 문제가 없다. 이때 컨테이너 안에 그 UID의 홈 디렉터리가 없어 일부 도구가 cache를 쓸 곳을 찾지 못할 수 있으므로, `HOME`이나 cache 경로를 쓰기 가능한 곳으로 지정한다.

공유 서버에서는 보안 문제도 있다. `docker` 그룹에 속한 사용자는 호스트의 아무 디렉터리나 root 권한으로 마운트할 수 있으므로, 사실상 root와 같은 권한을 가진다. 여러 사람이 쓰는 서버라면 rootless 모드의 Docker, Podman, 또는 다음 절의 Apptainer처럼 사용자 권한으로 실행되는 도구를 검토한다.

**shared memory(`--shm-size`).** PyTorch의 DataLoader는 `num_workers > 0`일 때 worker 프로세스가 만든 batch를 shared memory(`/dev/shm`)를 통해 주 프로세스로 넘긴다. Docker 컨테이너의 `/dev/shm` 기본 크기는 64 MiB다. worker마다 `prefetch_factor`개(기본값 2)의 batch를 미리 준비하므로, 대기 중인 batch가 차지하는 양을 어림할 수 있다.

```python {exec="browser" session="docker"}
B, C, H, W = 64, 3, 224, 224                    # 이미지 batch (B, C, H, W)
bytes_per_batch = B * C * H * W * 4             # float32
num_workers, prefetch_factor = 8, 2             # worker마다 미리 준비해 두는 batch 수
in_flight = num_workers * prefetch_factor

need_mib = in_flight * bytes_per_batch / 2**20
print(f"batch 하나: {bytes_per_batch / 2**20:.1f} MiB")
print(f"동시에 대기하는 batch: {in_flight}개 -> 약 {need_mib:.0f} MiB")
print(f"Docker 기본 /dev/shm 64 MiB의 {need_mib / 64:.1f}배")
```

```output
batch 하나: 36.8 MiB
동시에 대기하는 batch: 16개 -> 약 588 MiB
Docker 기본 /dev/shm 64 MiB의 9.2배
```

흔한 이미지 학습 설정 하나만으로 기본값의 9배가 필요하다. 이 어림은 label과 내부 오버헤드를 빼고 계산한 하한이다. 부족하면 worker가 `bus error`나 shared memory 관련 오류로 죽는다. `--shm-size=16g`처럼 넉넉히 늘리거나, 호스트의 IPC 네임스페이스를 공유하는 `--ipc=host`를 쓴다.

**네트워크와 포트.** 컨테이너 안의 Jupyter나 TensorBoard에 접속하려면 포트를 연결한다. `-p 8888:8888`은 호스트의 **모든 네트워크 인터페이스**에 포트를 연다. Docker가 추가하는 방화벽 규칙 때문에 호스트 방화벽 설정과 상관없이 외부에서 접근 가능해질 수 있다. 공유 서버에서는 `-p 127.0.0.1:8888:8888`로 localhost에만 열고, SSH 터널(2.3장)로 접속한다.

## 개발 컨테이너와 HPC

**devcontainer.** 편집기(VS Code 등)가 컨테이너 안에서 개발 환경을 여는 규격이다. 저장소의 `.devcontainer/devcontainer.json`에 사용할 Dockerfile과 실행 옵션을 적어 두면, 새 구성원이 저장소를 열 때 같은 컨테이너가 만들어진다(집필 시점 예시).

```json
{
  "name": "proj",
  "build": { "dockerfile": "../Dockerfile" },
  "runArgs": ["--gpus=all", "--shm-size=16g"],
  "mounts": ["source=/data/shared,target=/data,type=bind,readonly"]
}
```

연구용 Dockerfile 하나를 실행 image와 개발 환경에 함께 쓰면, "개발할 때와 실행할 때 환경이 다르다"는 문제가 줄어든다.

**Apptainer(Singularity).** 대부분의 HPC 클러스터는 보안상 Docker daemon을 허용하지 않고, 대신 Apptainer(이전 이름 Singularity)를 제공한다. Apptainer는 사용자 권한으로 컨테이너를 실행하고, image를 SIF라는 단일 파일로 다룬다. Docker image를 변환해 쓸 수 있으므로, Docker로 만든 연구 image를 클러스터에서 그대로 실행할 수 있다(집필 시점 예시. 클러스터의 안내 문서를 먼저 확인한다).

```bash
# registry의 Docker image를 SIF 파일로 변환
apptainer build proj.sif docker://registry.example.org/lab/proj@sha256:<digest>
# GPU(--nv)를 쓰고, 데이터 디렉터리를 연결해 실행
apptainer exec --nv --bind /scratch/me/data:/data proj.sif python train.py --data /data
```

Docker와 기본 동작이 다른 점이 있다. Apptainer는 기본적으로 사용자의 홈 디렉터리와 현재 디렉터리 등을 컨테이너에 자동으로 연결하고, 호스트 사용자 그대로 실행된다. 그래서 UID 문제는 없지만, 홈 디렉터리의 `~/.local`에 있는 패키지가 컨테이너 안의 Python에 섞여 들 수 있다(3.3장의 사용자 site-packages 문제). 컨테이너 안에서도 `python -m site`로 확인한다.

## 재현 가능한 빌드

Dockerfile이 같아도 나중에 다시 빌드한 image는 다를 수 있다. 시간에 따라 바뀌는 입력이 있기 때문이다.

- **base image tag**: `python:3.12-slim`은 보안 패치가 나올 때마다 새 image를 가리킨다.
- **시스템 패키지**: `apt-get update && apt-get install`은 빌드 시점의 최신 패키지를 받는다.
- **언어 패키지**: lockfile 없이 설치하면 빌드 시점의 최신 버전이 들어온다.
- **외부 다운로드**: `curl`로 받는 파일, `latest` tag의 도구 image(`ghcr.io/astral-sh/uv:latest` 등).

재현이 중요한 image는 이 입력들을 고정한다.

1. `FROM`과 `COPY --from`의 image를 digest로 고정한다: `FROM python:3.12-slim@sha256:<digest>`. tag를 함께 적어 두면 사람이 읽기 쉽다.
2. Python 패키지는 lockfile로 설치한다(`uv sync --locked`).
3. 시스템 패키지는 꼭 필요한 것만 설치하고, 가능하면 버전을 명시한다.
4. **빌드 기록**을 남긴다. 결과 image의 digest, 빌드한 git 커밋, 빌드 날짜를 실험 기록에 함께 적는다. 커밋 해시를 image에 label로 넣어 두면 image만 보고도 출처를 찾을 수 있다.

```bash
GIT_SHA=$(git rev-parse HEAD)
docker build --label org.opencontainers.image.revision="$GIT_SHA" -t lab/proj:"${GIT_SHA:0:8}" .
docker push lab/proj:"${GIT_SHA:0:8}"
docker image inspect --format '{{json .RepoDigests}}' lab/proj:"${GIT_SHA:0:8}"   # 기록할 digest
```

완벽한 비트 단위 재현까지는 필요하지 않은 경우가 많다. 연구에서 중요한 것은 **나중에 같은 image를 다시 실행할 수 있는가**이고, 그것은 빌드 결과를 digest로 registry에 보관하는 것으로 대부분 해결된다. 다시 빌드하는 능력은 그 image가 사라졌을 때를 위한 보험이다.

## 실습

연구 프로젝트의 Dockerfile을 작성하고, 로컬 가상환경과 컨테이너에서 같은 결과가 나오는지 확인한다. 실습 파일은 `labs/environments/docker/`에 있다.

- `Dockerfile`: 이 책의 실습 환경(`labs/pyproject.toml`, `labs/uv.lock`)으로 image를 만든다. lockfile을 먼저 복사하는 순서, uv cache mount, `Dockerfile.dockerignore`(BuildKit이 Dockerfile 옆에서 찾는 전용 ignore 파일)를 적용했다.
- `smoke_test.py`: 고정된 시드로 256 × 256 선형 시스템을 풀고, 결과를 `expected.json`의 값과 **허용 오차 안에서** 비교한다. BLAS 구현이나 CPU에 따라 마지막 자리가 달라질 수 있으므로 해시로 비교하지 않는다. 결과는 `OUTPUT_DIR`에 JSON으로 쓴다.

먼저 로컬 가상환경에서 실행한다. 집필 환경의 결과다.

```bash
uv run --project labs python labs/environments/docker/smoke_test.py
```

```output
python 3.13.18 / numpy 2.5.4 / x86_64
GPU: torch 없음
x[:3] = [ 1.220559  0.498932 -0.227598], |x| = 25.384727, 잔차 = 9.7e-13
결과 기록: outputs/smoke_result.json
SMOKE TEST 통과
```

다음으로 image를 빌드하고 컨테이너에서 실행한다(Docker가 있는 머신에서. 집필 환경에서는 실행하지 못했다).

```bash
docker build -f labs/environments/docker/Dockerfile -t lab/smoke:dev labs
mkdir -p outputs
docker run --rm --user "$(id -u):$(id -g)" \
    -v "$PWD/outputs:/outputs" -e OUTPUT_DIR=/outputs lab/smoke:dev
ls -l outputs/smoke_result.json       # 호스트에 남았는가, 소유자가 자신인가
```

확인할 것은 세 가지다. ① 컨테이너의 `SMOKE TEST`가 통과하는가. 이 image는 Python 3.12를 쓰므로 Python 버전 줄은 로컬과 다를 수 있다. `labs/uv.lock`에는 Python 3.12 이상용 NumPy가 따로 기록되어 있어 NumPy 버전은 같아야 한다. ② 결과 파일이 호스트의 `outputs/`에 남고 소유자가 자신인가. ③ `train.py`에 해당하는 파일 하나를 고쳐 다시 빌드할 때 의존성 설치 줄이 cache에서 재사용되는가(빌드 로그의 `CACHED` 표시). GPU 서버라면 자신의 프로젝트 image에 PyTorch를 넣고 `--gpus all`로 같은 확인을 한다.

이어서 연습문제를 푼다. image 참조 문자열을 registry·repository·tag·digest로 해석하는 함수를 작성한다.

::exercise{id="parse-image-ref"}

## 실패 사례

- **컨테이너 안에서 생성한 결과가 컨테이너 삭제와 함께 사라지는 것.** 마운트하지 않은 경로에 checkpoint를 저장하고 `--rm`으로 실행하면, 며칠짜리 학습 결과가 끝나는 순간 사라진다. 결과 경로는 항상 마운트하고, 학습을 시작하기 전에 짧은 실행으로 결과 파일이 호스트에 생기는지 확인한다.
- **DataLoader worker가 shared memory 부족으로 죽는 것.** 로컬에서 잘 돌던 코드가 컨테이너에서만 `bus error`나 worker 비정상 종료로 실패한다. 기본 `/dev/shm` 64 MiB가 원인인 경우가 많다. `--shm-size`나 `--ipc=host`를 실행 명령과 devcontainer 설정에 기본으로 넣는다.
- **image 안에 NVIDIA 드라이버를 설치하는 것.** 호스트 드라이버와 충돌하거나 버전 불일치 오류가 난다. 드라이버는 호스트의 것을 쓴다.
- **root로 실행해 호스트에 root 소유 파일을 남기는 것.** 결과 디렉터리를 일반 사용자가 지울 수 없게 되고, 관리자에게 부탁해야 한다. `--user`로 실행한다.
- **`latest` tag로 실험을 기록하는 것.** 몇 달 뒤 같은 이름으로 받은 image는 다른 image일 수 있다. digest를 기록한다.

## 통과 기준

- 새 사용자가 image 하나(digest로 지정)와 README의 명령 한 줄로 smoke test를 실행해 통과하고, 결과 파일이 호스트의 지정한 디렉터리에 자신의 소유로 남는다.
- 자신의 연구 프로젝트 Dockerfile이 의존성 기록을 코드보다 먼저 복사하고, 데이터는 읽기 전용 마운트로, 결과는 쓰기 가능한 마운트로 다룬다.
- 연습문제 `parse-image-ref`의 테스트를 통과한다.
