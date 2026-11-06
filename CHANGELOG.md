# Changelog

이 파일은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/) 형식을 따른다.

## [Unreleased]

## [0.1.0] - 2026-11-06 — 프로토타입 1

### Added
- 웹 사이트(`apps/web`, Astro 정적 사이트): 홈(레이어 스택), 부·장·트랙 페이지, 선수 지식 지도, 진도 페이지, ⌘K 검색, 라이트·다크 테마, floating dock
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
