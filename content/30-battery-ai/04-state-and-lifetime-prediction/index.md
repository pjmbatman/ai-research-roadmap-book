---
id: battery-ai/state-and-lifetime-prediction
title: "상태 추정과 수명 예측"
title_en: "State Estimation and Lifetime Prediction"
tier: core
status: outline
prerequisites: [battery-ai/battery-data, reliability/uncertainty-quantification]
---

# 상태 추정과 수명 예측

## 학습 목표

- SOC·SOH·RUL 문제를 엄밀히 정의한다
- 불확실성을 포함한 예측을 평가한다

## 문제 정의

- 입력 구간 · 정답 정의 · 예측 시점 · 일반화 대상

## SOC·SOH 추정

- 모델 기반 · 데이터 기반 · 하이브리드

## 수명 예측

- 조기 예측, RUL과 censoring, knee 예측

## 불확실성

- 예측 구간, 조건별 calibration

## 평가

- 새로운 셀 · 조건 · 데이터셋에서의 검증

## 실습

- 새로운 셀·조건에서 SOH와 예측 구간을 함께 출력하는 모델을 만들고 신뢰성을 평가한다.

## 실패 사례

- 실제로 측정할 수 없는 입력(예: 전체 용량 곡선)을 사용하는 것

## 통과 기준

- 실제 측정 가능한 입력만으로 예측하고 조건별 오차와 coverage를 보고한다.
