---
id: bayesian-inference/mcmc
title: "근사 추론 I: MCMC"
title_en: "Approximate Inference I: MCMC"
tier: advanced
status: outline
prerequisites: [bayesian-inference/bayesian-thinking, probability/stochastic-processes]
---

# 근사 추론 I: MCMC

## 학습 목표

- MCMC의 원리를 이해한다
- 주요 알고리즘을 사용한다
- 수렴을 진단한다

## 왜 근사가 필요한가

- 정규화 상수 계산의 어려움

## Metropolis–Hastings

- 제안분포, 수용 확률, detailed balance

## Gibbs sampling

- 조건부분포 샘플링

## HMC와 NUTS

- gradient를 이용한 탐색, 현대 확률 프로그래밍의 기본 엔진

## 수렴 진단

- trace plot · R-hat · ESS, divergence

## 실습

- Metropolis–Hastings를 구현하고 같은 모델을 확률 프로그래밍 도구(NUTS)로 풀어 진단 지표를 비교한다.

## 실패 사례

- 수렴 진단 없이 사후 표본을 사용하는 것

## 통과 기준

- MCMC 결과의 수렴 여부를 진단 지표로 판정한다.
