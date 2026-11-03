---
id: architectures/time-series-deep-learning
title: "시계열 딥러닝"
title_en: "Deep Learning for Time Series"
tier: core
status: outline
prerequisites: [architectures/recurrent-and-ssm, classical-ml/time-series-analysis]
---

# 시계열 딥러닝

## 학습 목표

- 시계열 딥러닝 모델의 계열을 안다
- 시계열 특유의 평가 함정을 피한다

## 모델 계열

- RNN · TCN · Transformer 계열 · MLP 기반 모델 · SSM

## 입력 설계

- window · stride · horizon, 공변량과 미래에 알려진 입력

## 불규칙·다변량 시계열

- 결측 · 다중 주기 · 채널 간 관계

## 시계열 Foundation Model

- 사전학습 시계열 모델의 현황과 평가

## 평가의 함정

- 단순 baseline 대비 개선, 데이터셋 선택 편향

## 실습

- 시계열 예측에서 최신 딥러닝 모델과 단순 선형 baseline을 같은 프로토콜로 비교한다.

## 실패 사례

- 단순 baseline 없이 복잡한 모델의 성능을 주장하는 것

## 통과 기준

- 시계열 과제에서 모델·입력 설계·평가 프로토콜을 정당화한다.
