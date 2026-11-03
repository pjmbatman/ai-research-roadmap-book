---
id: scientific-ml/neural-operators
title: "Neural Operator와 동역학 학습"
title_en: "Neural Operators and Learning Dynamics"
tier: advanced
status: outline
prerequisites: [scientific-ml/physics-and-learning, linear-algebra/function-spaces]
---

# Neural Operator와 동역학 학습

## 학습 목표

- 함수→함수 매핑으로 문제를 정의한다
- 주요 neural operator를 구현한다
- 해상도·형상 일반화와 장기 rollout을 평가한다

## operator learning

- 입력 함수에서 출력 함수로의 매핑, 이산화와의 관계

## DeepONet

- branch · trunk 구성

## FNO

- Fourier 공간의 적분 커널

## 다른 계열

- kernel integral · graph 기반 operator · Transformer 기반 operator

## 일반화

- mesh · 해상도 · 형상 일반화

## 시간 동역학

- one-step operator와 trajectory operator, autoregressive rollout의 장기 안정성

## 제약

- 미분 상태와 대수 제약의 처리

## 실습

- 같은 PDE 데이터셋에 DeepONet과 FNO를 학습해 해상도 변화와 장기 rollout에서 성능을 비교한다.

## 실패 사례

- 학습 해상도에서만 평가하고 해상도 불변성을 주장하는 것

## 통과 기준

- 어떤 입력 함수와 출력 함수 사이의 매핑을 어떤 이산화로 학습하는지 명시하고 평가한다.
