---
id: optimization/first-order-methods
title: "1차 최적화 방법"
title_en: "First-Order Methods"
tier: core
status: outline
prerequisites: [optimization/optimization-basics]
---

# 1차 최적화 방법

## 학습 목표

- gradient descent와 SGD의 수렴을 이해한다
- momentum의 역할을 설명한다

## Gradient descent

- 학습률과 곡률, 조건수에 따른 수렴 속도

## Line search

- Armijo 조건

## SGD

- mini-batch gradient의 잡음, 기대값과 분산, 수렴 조건

## Momentum

- heavy ball · Nesterov, 진동 억제

## 수렴 이론 개요

- 볼록·강볼록·비볼록에서의 수렴률

## 실습

- 조건수가 다른 이차함수에서 GD·momentum·SGD의 궤적을 시각화하고 학습률 범위를 실험한다.

## 실패 사례

- 학습률을 곡률과 무관한 상수로 생각하는 것

## 통과 기준

- 이차함수에서 GD의 최대 안정 학습률을 유도하고 실험으로 확인한다.
