---
id: research-landscape/first-pipeline
title: "첫 번째 연구 파이프라인 완주하기"
title_en: "Your First End-to-End Research Pipeline"
tier: core
status: outline
prerequisites: [research-landscape/problem-formulation]
---

# 첫 번째 연구 파이프라인 완주하기

## 학습 목표

- 데이터부터 결과 보고까지 최소 파이프라인을 한 번 완주한다
- 이후 각 부가 이 파이프라인의 어느 부분을 깊게 다루는지 안다

## 최소 파이프라인의 구성

- 문제 정의 → 데이터 로드 → 분할 → baseline → 신경망 → 평가 → 보고

## 한 번에 완주하기

- 공통 데이터셋, 단일 스크립트, 단일 프로세스
- forward · loss · backward · update의 첫 만남

## 의도적으로 생략한 것들

- 설정 관리, 재현성, 불확실성, 분산학습 — 각각 어느 부에서 다루는가

## 결과를 의심하는 첫 연습

- 너무 좋은 결과, 너무 나쁜 결과, 재실행 시 달라지는 결과

## 실습

- 제공된 공통 데이터셋으로 선형 baseline과 작은 MLP를 학습하고, 같은 분할·metric으로 비교한 한 쪽짜리 보고서를 쓴다.

## 실패 사례

- 처음부터 Docker·분산학습·추상화된 trainer를 도입하는 것
- baseline 없이 신경망부터 학습하는 것

## 통과 기준

- 다른 사람이 같은 명령으로 같은 숫자를 다시 얻는다.
