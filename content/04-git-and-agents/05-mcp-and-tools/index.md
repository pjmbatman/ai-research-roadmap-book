---
id: git-and-agents/mcp-and-tools
title: "MCP와 도구 연결"
title_en: "MCP and Tool Integration"
tier: advanced
status: outline
prerequisites: [git-and-agents/agent-instructions]
---

# MCP와 도구 연결

## 학습 목표

- MCP의 client·server·tool 구조를 이해한다
- 서버를 등록하고 권한을 관리한다
- 외부 데이터의 지시를 다루는 원칙을 안다

## MCP의 구조

- host · client · server, tool · resource · prompt

## 연결 방식

- 로컬 프로세스(stdio) 서버와 원격(HTTP) 서버

## 등록과 인증

- 설정 파일 위치와 범위(사용자·프로젝트), 환경변수, OAuth, 권한 범위

## 연구에 쓰는 도구 연결

- filesystem · Git · database · 문서 · 실험 추적 · 데이터 허브 연동

## 보안과 신뢰

- 도구 결과 검증, prompt injection, 외부 문서·데이터에 포함된 지시를 데이터로 다루기
- 제품별 등록 화면은 갱신되는 운영 문서로 관리

## 실습

- 연구 데이터베이스나 파일 저장소를 MCP 서버로 연결하고, 에이전트가 읽기 전용으로 질의하게 한다.

## 실패 사례

- 과도한 권한의 토큰을 등록하는 것
- 도구가 반환한 텍스트의 지시를 그대로 실행하는 것

## 통과 기준

- MCP 서버 하나를 최소 권한으로 등록하고 동작·실패 처리를 검증한다.
