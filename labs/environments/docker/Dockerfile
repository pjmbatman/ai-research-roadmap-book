# syntax=docker/dockerfile:1
# 3.5 실습용 Dockerfile. 빌드 context는 labs/ 디렉터리다(이 책의 실습 환경 pyproject.toml·uv.lock을 쓴다).
#
#   docker build -f labs/environments/docker/Dockerfile -t lab/smoke:dev labs
#   docker run --rm --user "$(id -u):$(id -g)" \
#       -v "$PWD/outputs:/outputs" -e OUTPUT_DIR=/outputs lab/smoke:dev
#
# 태그와 버전은 집필 시점 예시다. 재현이 필요하면 base image와 uv image를 digest로 고정한다(본문 '재현 가능한 빌드').

FROM python:3.12-slim

# uv 실행 파일만 공식 이미지에서 복사한다
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1

# 1) 의존성 기록만 먼저 복사하고 설치한다: 코드가 바뀌어도 이 layer는 cache에서 재사용된다
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# 2) 코드는 마지막에 복사한다
COPY environments/docker/smoke_test.py environments/docker/expected.json ./

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "smoke_test.py"]
