---
id: git-and-agents/ai-coding-tools
title: "AI 코딩 도구의 활용과 검증"
title_en: "Using and Verifying AI Coding Tools"
tier: core
status: outline
prerequisites: [git-and-agents/git-fundamentals]
---

# AI 코딩 도구의 활용과 검증

## 학습 목표

- 대화형·IDE형·에이전트형 도구의 차이를 안다
- AI가 만든 코드와 수식을 검증한다
- 생산성과 검증 비용의 균형을 잡는다

## 도구의 형태

- 대화형 assistant · IDE 자동완성 · 터미널·클라우드 코딩 에이전트

## 잘 맡길 수 있는 일

- 코드 탐색·설명 · 초안 · 리팩터링 · 테스트·문서·CLI 작성 · 오류 분석

## 검증해야 하는 것

- 존재하지 않는 API, 잘못된 수식, 가짜 인용, 조용히 바뀐 실험 조건
- 수식↔코드 대응, shape, gradient, 데이터 분할, metric

## 작업 단위와 프롬프트

- 작고 검증 가능한 단위로 요청하기, 맥락 제공, 완료 조건 명시

## 실습

- AI가 작성한 학습 코드를 shape·gradient·데이터 분할·metric 관점에서 리뷰하고 발견한 결함을 기록한다.

## 실패 사례

- 돌아가는 코드를 맞는 코드로 착각하는 것
- AI가 제안한 인용을 확인 없이 사용하는 것

## 통과 기준

- AI가 만든 변경을 리뷰해 연구 결론에 영향을 주는 결함을 찾아낸다.
