# 3.5 Docker와 컨테이너 — 실습

장: [content/03-environments/05-docker](../../../content/03-environments/05-docker/index.md)

| 파일 | 내용 |
|---|---|
| `cells.py` | 본문 코드 셀. 본문 output 블록은 이 파일의 실행 결과다 |
| `Dockerfile`, `Dockerfile.dockerignore` | 실습 환경(`labs/pyproject.toml`, `labs/uv.lock`)으로 smoke test image를 만든다 |
| `smoke_test.py`, `expected.json` | 로컬과 컨테이너에서 같은 결과가 나오는지 허용 오차로 확인한다 |
| `test_quiz.py` | 퀴즈 정답 검증 |
| `exercises/parse-image-ref/` | 연습문제: image 참조 해석하기 |

저장소 루트에서 실행한다.

```bash
# 로컬 가상환경
uv run --project labs python labs/environments/docker/smoke_test.py

# 컨테이너 (빌드 context는 labs/)
docker build -f labs/environments/docker/Dockerfile -t lab/smoke:dev labs
mkdir -p outputs
docker run --rm --user "$(id -u):$(id -g)" \
    -v "$PWD/outputs:/outputs" -e OUTPUT_DIR=/outputs lab/smoke:dev
```

집필 환경에는 Docker daemon이 없어 컨테이너 빌드·실행은 확인하지 못했다. 로컬 실행 결과만 본문에 실었다.
