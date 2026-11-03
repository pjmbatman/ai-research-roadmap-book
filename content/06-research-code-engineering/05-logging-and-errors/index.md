---
id: research-code-engineering/logging-and-errors
title: "로깅과 오류 처리"
title_en: "Logging and Error Handling"
tier: core
status: outline
prerequisites: [research-code-engineering/cli-design]
---

# 로깅과 오류 처리

## 학습 목표

- logging 모듈로 수준별 로그를 남긴다
- 실패를 명확하게 드러내는 코드를 작성한다

## logging

- logger · handler · level, 파일과 콘솔 동시 출력, run 디렉터리별 로그

## 무엇을 기록하는가

- 설정 · 환경 · 데이터 통계 · 진행 상황 · 경고

## 오류 처리 원칙

- fail loud: 조용히 넘어가는 except 금지, assert와 명시적 검증

## 경고의 관리

- warnings, 반복 경고 억제와 그 위험

## 실습

- print 기반 코드를 logging으로 바꾸고, 잘못된 입력에 즉시 실패하도록 검증을 추가한다.

## 실패 사례

- except: pass로 오류를 숨기는 것
- 로그에 남지 않은 설정으로 실험하는 것

## 통과 기준

- 실패한 실험의 로그만으로 원인을 찾는다.
