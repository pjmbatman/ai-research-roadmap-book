---
id: reinforcement-learning/model-based-control
title: "모델 기반 방법과 최적제어"
title_en: "Model-Based Methods and Optimal Control"
tier: advanced
status: outline
prerequisites: [reinforcement-learning/mdp-and-dp, optimization/constrained-optimization]
---

# 모델 기반 방법과 최적제어

## 학습 목표

- 최적제어와 강화학습의 관계를 이해한다
- MPC를 구현한다

## 최적제어

- LQR, 비용함수와 제약

## MPC

- receding horizon, 제약 처리

## 모델 기반 RL

- 학습된 동역학 모델로 계획하기, 모델 오차의 누적

## 미분가능 시뮬레이션과 제어

- gradient 기반 정책 최적화

## 실습

- 학습된 동역학 모델과 MPC로 제약이 있는 제어 문제를 풀고 실제 모델 대비 성능을 평가한다.

## 실패 사례

- 학습된 모델의 오차를 무시한 계획

## 통과 기준

- 같은 제어 문제를 MPC와 RL로 풀어 장단점을 비교한다.
