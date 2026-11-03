---
id: data/splitting-and-leakage
title: "데이터 분할과 누수 방지"
title_en: "Data Splitting and Leakage Prevention"
tier: core
status: outline
prerequisites: [data/preprocessing-and-eda]
---

# 데이터 분할과 누수 방지

## 학습 목표

- 일반화 대상에 맞는 분할을 설계한다
- 누수의 유형을 식별하고 방지한다
- test set을 위생적으로 관리한다

## 분할 전략

- random · group · time-based · stratified split, 일반화 대상과의 대응

## 독립성의 단위

- subject · cell · device · batch · 실험실 단위

## 누수의 유형

- 중첩 window와 인접 시점, 미래 정보, 전처리·feature selection·튜닝에서의 누수, 중복 샘플

## fit은 학습 데이터에서만

- 전처리 통계량 추정과 적용, pipeline 패턴

## validation과 test의 역할

- 모델 선택과 최종 보고의 분리, test set 사용 기록

## 사전학습 데이터와의 중복

- pretraining과 downstream 사이의 중복 점검

## 외부 검증

- 새 도메인 · 새 조건 · 새 시점에서의 검증

## 실습

- 같은 데이터에 random split과 group split을 적용해 성능 차이를 측정하고, 누수 감사(leakage audit) 체크리스트를 작성한다.

## 실패 사례

- 같은 셀의 다른 사이클이 학습·테스트에 섞이는 것
- test 성능을 보고 하이퍼파라미터를 고르는 것

## 통과 기준

- 자신의 분할이 주장하려는 일반화 대상과 일치함을 논증하고 누수 테스트를 통과한다.
