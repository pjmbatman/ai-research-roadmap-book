---
id: research-code-engineering/cli-design
title: "CLI 설계"
title_en: "Designing Command-Line Interfaces"
tier: core
status: outline
prerequisites: [research-code-engineering/configuration]
---

# CLI 설계

## 학습 목표

- argparse로 모듈러 CLI를 만든다
- 설정 파일과 명령행 인자를 일관되게 병합한다

## argparse

- 위치 인자 · 옵션 · 타입 · 기본값 · help

## subcommand

- train · eval · predict · export를 하나의 CLI로

## 다른 도구

- typer · click의 위치

## 설정과 CLI의 병합

- 우선순위: 기본값 < 설정 파일 < 명령행

## 진입점 등록

- pyproject의 scripts, python -m 실행

## 실습

- 프로젝트에 train/eval subcommand를 가진 CLI를 만들고 --help만으로 사용법을 알 수 있게 한다.

## 실패 사례

- 인자 이름과 설정 키가 달라 혼란을 주는 것
- 잘못된 인자를 조용히 무시하는 것

## 통과 기준

- 동료가 문서 없이 --help만 보고 실험을 실행한다.
