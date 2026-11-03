# 실습 코드 (labs/)

각 장의 `## 실습` 절에 대응하는 코드를 둔다.

## 경로 규칙

```
labs/<부 slug>/<장 slug>/
├── README.md        # 목표 · 실행 방법 · 기대 결과 · 대응하는 장 링크
├── *.py             # 검증 대상 코드
└── *.ipynb          # (선택) 탐색·시각화
```

- 번호 없이 slug만 쓴다. 예: `labs/linear-algebra/affine-and-nonlinear/`
- 데이터는 커밋하지 않는다. 내려받기 스크립트 또는 공통 데이터셋 경로를 README에 적는다.
- 공통 Python 환경(`pyproject.toml` + lockfile)은 첫 실습을 작성할 때 이 디렉터리에 추가한다.
  이 환경이 웹 사이트의 로컬 런타임이 된다: 학습자가 런처 명령 하나로 자기 컴퓨터에 커널을 띄우면 사이트가 연결해 `exec="python"` 코드를 실행한다.
- 자동 채점 연습문제: `exercises/<id>/`(problem.md · starter.py · test_exercise.py · solution.py · exercise.toml). 형식은 `editorial/conventions.md` §8.
