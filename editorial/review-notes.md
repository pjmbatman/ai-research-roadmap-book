# 초고 리뷰 메모 (v0.4 전권 초고)

> 2026-11-07. 215개 장을 `status: draft`로 쓴 뒤 남긴 리뷰 안내다. 장을 `review`·`done`으로 올리기 전에 아래를 확인한다.
> 장을 리뷰할 때는 `.claude/skills/review-chapter/SKILL.md` 절차를 따른다.

## 1. 자동으로 확인된 것

- `python3 scripts/book.py check` — 구조·frontmatter·링크·코드 셀(cells.py와 본문 일치)·참고문헌 항목 형식.
- `cd labs && uv run pytest -q .` — 해답 코드, 퀴즈 정답(test_quiz.py).
- `EXERCISE_MODULE=starter` — 연습문제마다 시작 코드는 테스트에 실패한다.
- `cd apps/web && npm run test:cells` — 본문의 브라우저 코드 셀을 Pyodide에서 실행해 output 블록과 비교한다. 로컬 CPython과 Pyodide(32비트 wasm, 다른 numpy·scipy·pandas·scikit-learn 버전)의 출력이 같도록 반올림 잡음·경계 판정·버전별 문구는 판정 형태로 찍었다.
- `exec="python"`(PyTorch) 셀은 집필 시 CPU에서 실행한 결과다. CI는 실행하지 않는다.

## 2. 사람이 확인해야 하는 것

### 2.1 인용과 수치
집필 환경에서 arXiv·Crossref·출판사 사이트 접근이 막혀 있었다. 대부분의 서지는 웹 검색 결과(서지 기록·초록)로 확인했고, 원문을 직접 읽은 것은 아니다. `data/references.bib`의 `note`에 확인 근거를 적었다.

- **원문 대조 우선순위가 높은 것**: 본문에 수치를 인용한 문헌(예: Severson 2019의 셀 수·오차, Burger 2020의 실험 횟수, GraphCast 비교 수치), 교재 장·절 번호(Sutton–Barto, Bishop, Rasmussen–Williams).
- **인용 없이 일반 서술로 둔 것**: 각 장의 `:::note[분야 예시]` 일부, 표준 결과 몇 가지(conformal 포함률의 Beta 분포 등). 리뷰 때 출처를 붙이거나 서술을 낮춘다.
- **참고 문헌 절이 없는 장**: 2부, 3부, 6.1–6.4, 7.1–7.10, 8부, 17.6–17.10 등 원문을 확인하지 못한 장. 후보 문헌은 각 장 커밋 메시지·에이전트 보고에 있었던 것을 아래에 모았다.
  - 7부: Eckart–Young 1936, Halko–Martinsson–Tropp 2011, Kolda–Bader 2009, Tenenbaum 2000(Isomap), Zaheer 2017(Deep Sets), Cohen–Welling 2016
  - 8부: Baydin 외 2018, Griewank–Walther 2008, Papamakarios 외 2021
  - 17부: Ioffe–Szegedy 2015, Srivastava 외 2014, Paszke 외 2019, Ansel 외 2024
  - 22.5–22.7: ZeRO(Rajbhandari 2020), activation checkpointing(Chen 2016), ring all-reduce(Patarasuk–Yuan 2009), Jacob 2018(int8)
- **기억에 의존한 도구 사실**: 2부(bash 시작 파일 순서, ssh_config 규칙, Slurm 신호·requeue), 3부(Ubuntu 드라이버 명령, PEP 751 상태). 모두 "집필 시점 예시"로 표시했지만 man 페이지·공식 문서로 다시 확인한다.

### 2.2 장 사이 일관성
- 19.5(확산)와 19.6(flow matching)의 시간 규약(t=0이 데이터인지 잡음인지)과 확률 흐름 ODE 표기.
- 14.1 `D_KL` 대 부록 A `KL`, 9.7의 확산 계수 α 대 부록 A κ, "reparameterization trick"(11.6)과 "재매개변수화 기법"(13.6)의 번역 통일.
- 31.1·31.3이 26–30부를 장 번호로 참조한다. 26–30부 리뷰 뒤 다시 맞춘다.
- 학습 목표를 목차보다 늘린 장(10.7, 13.1–13.4, 24.x, 25.x 등): 목표가 늘어난 것이 장의 범위에 맞는지.

### 2.3 분량
권장 분량(180–320줄)을 넘는 장이 많다(최대 약 470줄). 수업 시트가 쪽으로 나누므로 읽기에는 문제가 없지만, 리뷰 때 줄일 곳을 찾는다.

### 2.4 브라우저 실행 환경
- 16.5–16.8의 일부 셀은 Pyodide에서 scikit-learn이 내는 threadpoolctl 경고를 숨기는 줄을 본문에 갖고 있다. 실행기(`py.worker.ts`)에서 경고를 거르면 이 줄을 뺄 수 있다.
- `np.linalg.cholesky`는 Pyodide에서 양의 정부호가 아닌 행렬에 오류 대신 NaN을 낸다. 판정에 쓰는 셀은 `scipy.linalg.cholesky`를 쓴다(7.5, 11.4).

## 3. 위젯

`data/widgets.toml`의 120개 위젯이 모두 구현되어(`status = "prototype"`) 각 장의 해당 절에 들어 있다. 위젯을 만들 때 계산부를 Node로 돌려 NumPy·SciPy 또는 장의 출력과 대조했다. 리뷰 때 확인할 것:

- **장 본문과 수치가 다른 위젯**: 위젯이 데이터를 새로 만들거나 자체 난수를 써서 장의 예시 숫자와 다르다(예: robust-outlier 6/10 대 본문 4/10, knn-rbf-scale 2D R², tree-boosting-overfit 시험 RMSE, bandit-regret·lqr-mpc·lora-rank 일부). 메시지는 같지만 숫자를 맞출지 정한다.
- **위젯이 드러낸 본문 확인 사항**:
  - 18.7: 무작위 분할에서 1-최근접 이웃의 이득을 "한 시간 어긋난 train window"로 설명하지만, L=48에서 1-최근접 이웃이 겹치는 window를 고르는 비율은 약 26%(L=96에서 약 67%)다.
  - 16.3: 표준화하지 않은 RBF도 길이척도가 크면 시험 R²가 0.95까지 오른다. "표준화하지 않으면 실패한다"는 서술의 강도를 확인한다.
  - 18.6 over-smoothing 위젯의 잔차 연결은 initial residual `(1−α)ÂH + αX`(α=0.2)로 구현했다.
- **수업 시트 배치**: 넓은 화면(960px 이상)에서는 시트를 창 높이까지 늘리고 조작·readouts 판을 그림 오른쪽에 둔다. 좁은 화면에서는 시트 안에서 스크롤한다.
- 위젯 모음 페이지 `/widgets/`에서 한 번에 조작해 볼 수 있다.
