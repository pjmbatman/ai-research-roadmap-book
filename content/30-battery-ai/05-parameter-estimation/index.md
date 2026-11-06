---
id: battery-ai/parameter-estimation
title: "파라미터 추정과 하이브리드 모델링"
title_en: "Parameter Estimation and Hybrid Modeling"
tier: advanced
status: outline
prerequisites: [battery-ai/battery-models, scientific-ml/system-identification-and-estimation]
---

# 파라미터 추정과 하이브리드 모델링

## 학습 목표

- 배터리 모델 파라미터를 추정한다
- 물리–데이터 하이브리드 모델을 구성한다

## 파라미터 추정

- 최적화 · 베이지안 추정, 식별 가능성

## 대리모델

- 전기화학 모델의 neural surrogate, 물성·운전 조건 일반화

## 하이브리드 모델

- 물리 모델 + 데이터 기반 보정, 열화 모델링

## 평가

- 정확도 · 물리적 일관성 · 속도

## 실습

- 물성·운전 조건 변화에 대한 배터리 동역학 대리모델을 만들고 정확도·물리적 일관성·속도를 함께 평가한다.

## 실패 사례

- 식별 불가능한 파라미터를 개별적으로 해석하는 것

## 통과 기준

- 대리모델을 고충실도 모델 대비 정확도·속도로 검증한다.
