---
id: git-and-agents/agent-instructions
title: "에이전트 지침: AGENTS.md·CLAUDE.md·SKILL.md"
title_en: "Agent Instructions: AGENTS.md, CLAUDE.md, SKILL.md"
tier: core
status: outline
prerequisites: [git-and-agents/ai-coding-tools]
---

# 에이전트 지침: AGENTS.md·CLAUDE.md·SKILL.md

## 학습 목표

- 프로젝트 지침 파일의 역할과 적용 범위를 안다
- 재사용 가능한 skill을 작성한다
- 지침을 버전 관리한다

## 프로젝트 지침 파일

- AGENTS.md · CLAUDE.md 등 도구별 지침, 전역·프로젝트·디렉터리 수준과 우선순위

## 무엇을 쓰는가

- 프로젝트 구조, 설치·실행·테스트 명령, 금지사항, 검증 기준, 데이터 취급 규칙

## Skill

- SKILL.md로 반복 절차를 패키징하기, 설명(description)과 트리거, 보조 스크립트

## 지침의 유지보수

- 지침도 코드처럼 리뷰·버전 관리, 오래된 지침 정리, 지침과 실제 코드의 불일치

## 실습

- 새 에이전트가 프로젝트를 탐색하고 최소 실험을 실행할 수 있는 AGENTS.md와 반복 작업 하나에 대한 SKILL.md를 작성한다.

## 실패 사례

- 지침을 너무 길게 써서 핵심 규칙이 묻히는 것
- 실제와 다른 명령이 지침에 남아 있는 것

## 통과 기준

- 처음 보는 에이전트가 지침만으로 smoke test를 통과한다.
