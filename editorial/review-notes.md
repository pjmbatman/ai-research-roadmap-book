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

## 3. 위젯 후보

구현된 위젯은 `data/widgets.toml`의 `prototype`이다. 집필 에이전트가 제안한 후보는 아래와 같다. 만들기로 하면 `status = "planned"`로 등록하고 장에 넣는다.

| 부 | 후보 |
|---|---|
| 1–6 | 분할 단위·예측 구간(1.3), seed 변동(1.5), tmux 구조·SSH 터널 방향(2부), CUDA 버전 계층·Docker 캐시(3부), commit DAG·선택 편향(4부), broadcasting·stride·roofline(5부), 설정 병합·argparse(6부) |
| 7–10 | 고유벡터·사영(7부), einsum 축·slerp·등변성 검사(7.7–7.10), 기울기·Hessian 등고선(8.1), Newton 끌개·반복법 경주·조건수 타원(9.1–9.4), Runge 보간·CFL(9.5–9.7), 손실 단면·1D 베이지안 최적화(10부) |
| 11–16 | 기저율(11.3), CLT·중요도 가중·Metropolis(11.5–11.7), 신뢰구간 포함률·부트스트랩·잔차 진단(12부), 부분 풀링·GMM EM(13부), KL 방향·double descent·MI 편향(14부), 실험 설계 커버리지·window 누수·robust 이상치(15부), 정규화 경로·트리 과적합·t-SNE perplexity·KM 곡선(16부) |
| 17–22 | smoothing 착시·학습 곡선 진단(17부), receptive field·SSM 커널·temperature·over-smoothing(18부), GAN 동역학·coupling flow·VAE 붕괴·flow 경로·생성 지표 실패·LoRA 랭크(19부), gridworld·밴딧 후회·LQR/MPC(20부), conformal 포함률·비용 임계값(21부), checkpointing·ring all-reduce·배치 지연(22부) |
| 23–31 | 인용 그래프·검정력·요인 설계·오차막대·축 왜곡·컬러맵(23부), PINN 손실 균형·스펙트럼 합성곱·칼만 필터·L-curve(24부), ABC 허용 오차·EIG·Pareto(25부), 격자 해상도·POD·closure·빔 Pareto(26부), 위도 가중·spread-skill·에너지 스펙트럼(27부), 상동성 분할·Kabsch 정렬·배치 보정(29부), SOC 필터·충전 Pareto(30부), 역량 레이더(31부) |
