---
id: earth-and-climate/weather-forecasting
title: "데이터 기반 기상 예측"
title_en: "Data-Driven Weather Forecasting"
tier: advanced
status: outline
prerequisites: [earth-and-climate/earth-system-data, architectures/time-series-deep-learning]
---

# 데이터 기반 기상 예측

## 학습 목표

- 데이터 기반 예보 모델의 구조를 이해한다
- 결정론적 예보와 확률적 예보를 구분한다

## 문제 정의

- 대기 상태에서 다음 상태로의 매핑, 리드 타임, autoregressive rollout

## 모델 구조

- 구면 위의 그래프·Transformer·neural operator

## 확률적 예보

- 앙상블, 생성모델 기반 예보

## 수치 예보와의 관계

- 초기조건(자료동화), 하이브리드 접근

## 실습

- 저해상도 재분석 데이터로 작은 autoregressive 예보 모델을 학습하고 리드 타임별 오차를 지속성(persistence)·기후값 baseline과 비교한다.

## 실패 사례

- 지속성·기후값 baseline 없이 예보 성능을 보고하는 것

## 통과 기준

- 리드 타임별 성능을 baseline과 함께 보고한다.
