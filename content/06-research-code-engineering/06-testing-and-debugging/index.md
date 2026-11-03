---
id: research-code-engineering/testing-and-debugging
title: "테스트와 디버깅"
title_en: "Testing and Debugging"
tier: core
status: outline
prerequisites: [research-code-engineering/logging-and-errors, git-and-agents/github-collaboration]
---

# 테스트와 디버깅

## 학습 목표

- 연구 결론을 바꿀 수 있는 실패를 우선 테스트한다
- 디버거로 버그를 추적한다
- CI로 최소 검증을 자동화한다

## 테스트의 종류

- unit · integration · smoke test, pytest 기초(fixture · parametrize)

## 연구 코드에서 테스트할 것

- shape · 단위 · 데이터 분할 · metric · checkpoint 복원의 불변조건
- 데이터 누수 테스트, 작은 데이터 과적합 테스트

## 디버깅

- stack trace 읽기, pdb · breakpoint() · IDE debugger, 원격 디버깅

## 체계적 디버깅

- 가설 세우기, 이분 탐색(git bisect), 최소 재현

## CI

- GitHub Actions로 설치·lint·테스트 자동 실행

## 실습

- 데이터 분할 누수·metric 집계 오류·checkpoint 복원 실패를 잡는 테스트를 작성하고 CI에 연결한다.

## 실패 사례

- 테스트하기 쉬운 것만 테스트하는 것
- 무작위성 때문에 불안정한 테스트

## 통과 기준

- 일부러 넣은 세 가지 버그를 테스트가 모두 잡아낸다.
