---
id: generative-and-foundation-models/flow-matching
title: "Flow Matching과 연속시간 생성모델"
title_en: "Flow Matching and Continuous-Time Generative Models"
tier: advanced
status: outline
prerequisites: [generative-and-foundation-models/diffusion-models, numerical-methods/ode-solvers]
---

# Flow Matching과 연속시간 생성모델

## 학습 목표

- 속도장을 학습하는 생성모델을 이해한다
- 확산 모델과의 관계를 설명한다

## 연속 정규화 흐름

- 속도장과 ODE, 확률 경로

## flow matching

- 조건부 경로, 단순한 회귀 목적함수

## rectified flow와 최적 수송

- 직선 경로, 적은 step 생성

## 확산 모델과의 관계

- 같은 틀의 다른 선택

## 실습

- flow matching 모델을 구현하고 확산 모델과 학습 안정성·샘플링 비용을 비교한다.

## 실패 사례

- ODE solver 오차를 모델 오차로 오해하는 것

## 통과 기준

- flow matching의 목적함수를 유도하고 확산 모델과의 관계를 설명한다.
