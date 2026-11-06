---
id: battery-ai/charging-optimization
title: "충전 최적화와 제어"
title_en: "Charging Optimization and Control"
tier: advanced
status: outline
prerequisites: [battery-ai/battery-models, reinforcement-learning/model-based-control]
---

# 충전 최적화와 제어

## 학습 목표

- 충전 프로토콜 최적화 문제를 정의한다
- 안전 제약을 포함한 최적화를 수행한다

## 문제 정의

- 충전 시간 · 열화 · 안전의 다목적 최적화

## 방법

- 베이지안 최적화 · MPC · 강화학습

## 실험 설계

- 비싼 실험의 순차적 선택, 조기 예측과의 결합

## sim-to-real

- 시뮬레이션 최적화 결과의 실제 검증

## 안전 제약

- 전압 · 온도 · 리튬 석출 제약

## 실습

- 시뮬레이터에서 베이지안 최적화로 다단계 충전 프로토콜을 최적화하고 제약 위반 여부를 평가한다.

## 실패 사례

- 시뮬레이터의 최적해를 실제 셀에서 검증 없이 주장하는 것

## 통과 기준

- 다목적 최적화 결과를 Pareto front와 제약 만족 여부로 보고한다.
