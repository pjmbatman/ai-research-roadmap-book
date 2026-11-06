# Changelog

이 파일은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/) 형식을 따른다.

## [Unreleased]

### Changed
- **UI/UX를 처음부터 다시 만들었다** ([ADR 0010](editorial/decisions/0010-reference-based-design.md)). 이전 방향(Midnight·Glacier·Coral 테마, floating dock, 효과 도구 상자, 위젯 `kind`, 캔버스 지도)과 그 명세를 폐기했다.
  - 시각 기반: shadcn/ui v4 neutral 토큰을 그대로 쓴다. 상태 색은 Tailwind 팔레트에서 가져왔다.
  - 공통 머리: shadcn/ui 문서 사이트 형식. dock은 없앴다.
  - 홈·트랙·로드맵: roadmap.sh 형식. 부는 주제 노드, 장은 하위 노드이고, 노드 서랍에서 학습 중·완료·건너뜀을 고른다.
  - 로드맵 전체: 레이어별 트랙과 의존선, 트랙 서랍.
  - 장 읽기: Brilliant·Duolingo식 레슨 화면(레슨 바·진행 막대·하단 이전/계속), 퀴즈는 카드형 선택지와 주 버튼 하나.
  - 연습문제: TutorialKit·svelte식 분할 화면.
  - 위젯: Card 틀, 색은 테마 토큰을 따르고 초기화 버튼이 있다.
  - 코드 편집기: 토큰 기반 색과 GitHub 기본 문법 색.
- 위젯 레지스트리에서 디자인 명세 필드(`kind`·`interaction`·`binds`·`motion`)를 지웠다.
- 장 상태 모델: 시작 전 · 학습 중 · 완료 · 건너뜀 (roadmap.sh)

### Added
- UI/UX 벤치마킹(`editorial/benchmarks.md`)

## [0.1.0] - 2026-11-06 — 프로토타입 1

### Added
- 웹 사이트(`apps/web`, Astro 정적 사이트): 홈(레이어 스택), 부·장·트랙 페이지, 선수 지식 지도, 진도 페이지, ⌘K 검색, 라이트·다크 테마, floating dock
- 넘겨 읽기(ADR 0009): 장은 절 단위 단계, 퀴즈는 한 문항씩·대화상자, 연습문제·목록은 탭, 로딩 막대와 전환 애니메이션
- 장 페이지: 고정 절 박스, KaTeX 수식, 개념 퀴즈(단일·복수·수치+슬라이더, 학습 목표별 요약), 브라우저 Python 코드 셀(CodeMirror + Pyodide), 연습문제 자동 채점, 통과 기준 체크, 브라우저 진도 저장·내보내기
- 위젯 3종(prototype): `linear-map-2d`, `relu-regions`, `loss-to-estimator`
- 시범 장 초고: 7.2 아핀 구조와 비선형성, 12.3 분포 기반 문제 정의 (실습 스크립트, 연습문제 4개, 퀴즈 13문항)
- 부록 A 표기법 초고, 용어집 12개 항목 추가
- `book.py check`: 본문 실행 코드 블록과 `labs/**/cells.py`의 일치 검사
- CI: 실습 테스트·퀴즈 정답 계산 검증, 사이트 빌드·링크 검사·Playwright 종단 테스트. GitHub Pages 배포 워크플로
- 장 상태 `spec`(명세), v1 계획(`editorial/plans/v1-prototype.md`), ADR 0008(잠정)
- 저장소 구조, 컨벤션, 문체 가이드, 작업 흐름(`editorial/workflow.md`)과 결정 기록(ADR 0001–0006)
- 7개 레이어 · 12개 트랙 · 33개 부 · 215개 장의 상세 목차(`OUTLINE.md`)
- `scripts/book.py`: 구조·선수 지식·레이어 방향·퀴즈·삽입 요소 검증, 목차 생성
- 개념 퀴즈 형식과 예시 4개 장(17문항), 위젯 레지스트리(18개 계획)
- 플랫폼 설계: 로그인 없는 정적 사이트 + 런타임 어댑터
- Claude Code skill: `write-chapter`, `review-chapter`, `add-quiz`
- 이슈·PR 템플릿
- 라이선스: 코드 MIT(`LICENSE`), 본문·데이터 CC BY 4.0(`LICENSE-CONTENT`)
- 디자인 시스템(`editorial/design-system.md`, ADR 0007): 단일 테마(Midnight · Glacier + Coral), 효과 도구 상자, 슬라이더·직접 조작·퀴즈 인터랙션 원칙
- 위젯 `kind`(manipulate · stepper · simulate · compare)와 수치 퀴즈의 슬라이더 응답(`slider`)
