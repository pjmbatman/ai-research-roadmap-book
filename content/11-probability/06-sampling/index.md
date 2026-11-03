---
id: probability/sampling
title: "Monte Carlo와 샘플링"
title_en: "Monte Carlo and Sampling"
tier: core
status: outline
prerequisites: [probability/expectation-and-limits]
---

# Monte Carlo와 샘플링

## 학습 목표

- Monte Carlo 추정과 분산을 이해한다
- 기본 샘플링 기법을 구현한다

## Monte Carlo 추정

- 기댓값 추정, 오차의 크기(1/√N)

## 기본 샘플링

- inverse CDF, rejection sampling, Box–Muller

## importance sampling

- 가중치, 유효 표본 크기, 분산 폭발

## 분산 감소

- control variate, antithetic, stratified sampling

## 재매개변수화

- reparameterization trick과 score function(REINFORCE) 추정량

## 실습

- 같은 기댓값을 naive MC·importance sampling·control variate로 추정해 분산을 비교한다.

## 실패 사례

- importance weight의 분산이 폭발하는 것을 놓치는 것

## 통과 기준

- 재매개변수화와 score function 추정량의 차이를 실험으로 보인다.
