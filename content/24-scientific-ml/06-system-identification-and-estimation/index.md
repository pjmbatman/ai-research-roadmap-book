---
id: scientific-ml/system-identification-and-estimation
title: "시스템 식별과 상태 추정"
title_en: "System Identification and State Estimation"
tier: core
status: outline
prerequisites: [scientific-ml/physics-and-learning, probability/multivariate-gaussian]
---

# 시스템 식별과 상태 추정

## 학습 목표

- 식별 가능성과 관측 가능성을 판단한다
- 필터로 상태를 추정한다

## 파라미터 식별

- 식별 가능성 · 민감도 · 실험 설계

## 관측 가능성

- 측정으로 상태를 알 수 있는가

## Kalman filter 계열

- KF · EKF · UKF, 잡음 공분산 튜닝

## particle filter

- 비선형·비가우시안 상태 추정

## 학습과 결합

- 학습된 동역학 + 필터, differentiable filter

## 실습

- 잡음이 있는 측정으로 EKF와 particle filter를 구현해 상태 추정 정확도와 불확실성을 비교한다.

## 실패 사례

- 식별 불가능한 파라미터를 추정하고 값을 해석하는 것

## 통과 기준

- 민감도 분석으로 식별 가능한 파라미터 집합을 정하고 추정한다.
