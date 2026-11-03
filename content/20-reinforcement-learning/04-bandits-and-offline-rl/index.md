---
id: reinforcement-learning/bandits-and-offline-rl
title: "밴딧·오프라인 RL·안전한 의사결정"
title_en: "Bandits, Offline RL, and Safe Decisions"
tier: advanced
status: outline
prerequisites: [reinforcement-learning/model-free-rl]
---

# 밴딧·오프라인 RL·안전한 의사결정

## 학습 목표

- 탐색–활용 trade-off를 이해한다
- 고정된 데이터로 정책을 학습·평가한다
- 안전 제약을 다룬다

## 밴딧

- ε-greedy · UCB · Thompson sampling, 실험 선택에의 응용

## 오프라인 RL

- 분포 이동 문제, 보수적 방법

## 오프 정책 평가

- importance sampling 기반 평가

## 안전 제약

- 제약 MDP, 안전한 탐색

## 실습

- Thompson sampling으로 실험 조건 선택 문제를 풀고 random 선택 대비 효율을 비교한다.

## 실패 사례

- 오프라인 데이터 밖의 행동을 과대평가하는 것

## 통과 기준

- 실제 실험 비용이 드는 문제에서 탐색 전략을 설계한다.
