---
id: chemistry-and-materials/property-prediction
title: "물성 예측"
title_en: "Property Prediction"
tier: core
status: outline
prerequisites: [chemistry-and-materials/molecular-representations, reliability/uncertainty-quantification]
---

# 물성 예측

## 학습 목표

- 분자·재료 물성 예측 모델을 학습하고 평가한다
- 화학 공간에서의 일반화를 검증한다

## 데이터

- 실험 데이터와 계산(DFT) 데이터, 측정 조건과 잡음

## 분할

- 무작위 · scaffold · 군집 · 시간 분할, 화학 공간의 외삽

## 모델

- 기술자 기반 회귀, GNN, 사전학습 모델

## 불확실성

- 적용 영역(applicability domain), 불확실성 기반 후보 선별

## 실습

- 같은 데이터에 무작위 분할과 scaffold 분할을 적용해 성능 차이를 측정하고 불확실성과 오차의 관계를 확인한다.

## 실패 사례

- 무작위 분할 성능으로 새로운 화학 구조에 대한 일반화를 주장하는 것

## 통과 기준

- 화학 공간 기반 분할로 평가한 결과와 적용 영역을 보고한다.
