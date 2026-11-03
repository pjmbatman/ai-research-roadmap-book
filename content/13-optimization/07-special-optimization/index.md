---
id: optimization/special-optimization
title: "특수한 최적화 문제"
title_en: "Special Optimization Problems"
tier: advanced
status: outline
prerequisites: [optimization/constrained-optimization]
---

# 특수한 최적화 문제

## 학습 목표

- 다목적·이중수준·블랙박스 최적화를 정의하고 푼다

## 다목적 최적화

- Pareto front, scalarization

## 이중수준 최적화

- 하이퍼파라미터 최적화, meta-learning의 구조

## 블랙박스 최적화

- random search · grid search · 진화 전략 · CMA-ES

## 베이지안 최적화

- 대리모델(GP) + 획득함수(EI · UCB), 실험 설계와의 연결

## 미분 불가능·이산 문제

- 정수 계획, relaxation의 개념

## 실습

- 비싼 시뮬레이터 함수를 베이지안 최적화와 random search로 최적화해 평가 횟수 대비 성능을 비교한다.

## 실패 사례

- 다목적 문제를 가중합 하나로 풀고 trade-off를 숨기는 것

## 통과 기준

- 실험 비용이 큰 문제에 베이지안 최적화를 적용하고 결과를 해석한다.
