---
id: research-code-engineering/naming-and-style
title: "네이밍·코딩 컨벤션"
title_en: "Naming and Coding Conventions"
tier: core
status: outline
prerequisites: [research-code-engineering/project-structure]
---

# 네이밍·코딩 컨벤션

## 학습 목표

- 일관된 이름 규칙을 적용한다
- shape·단위·시간 기준을 이름에 드러낸다
- 자동 도구로 스타일을 강제한다

## 이름 규칙

- 변수 · 함수 · 클래스 · 모듈 · 파일 · 디렉터리 · 실험 이름

## 의미를 드러내는 이름

- shape 주석과 접미사, 단위 접미사(_v · _a · _s), 시간 기준(t0 · horizon)

## 코드 스타일

- PEP 8, 함수 길이, 주석과 docstring 규칙

## 자동화 도구

- formatter · linter(ruff 등) · type checker, pre-commit

## 연구실 스타일 가이드

- 규칙 문서화와 예외 처리

## 실습

- 기존 코드에 연구실 네이밍 규칙과 lint를 적용하고, 이름 변경만으로 발견된 버그를 기록한다.

## 실패 사례

- x, x2, tmp, final_final 같은 이름
- 단위가 섞인 변수

## 통과 기준

- 프로젝트에 formatter·linter·pre-commit을 설정하고 CI에서 통과한다.
