---
id: bayesian-inference/variational-inference
title: "근사 추론 II: 변분추론과 Laplace 근사"
title_en: "Approximate Inference II: Variational Inference and Laplace"
tier: advanced
status: outline
prerequisites: [bayesian-inference/latent-variables-and-em]
---

# 근사 추론 II: 변분추론과 Laplace 근사

## 학습 목표

- ELBO를 유도한다
- 변분추론을 최적화 문제로 푼다
- Laplace 근사를 사용한다

## 변분추론

- KL 최소화와 ELBO, mean-field 근사

## 확률적 변분추론

- 재매개변수화, mini-batch

## amortized inference

- encoder로 사후를 근사하기(VAE와 연결)

## Laplace 근사

- 사후 최빈값 주변의 가우시안 근사, Hessian

## 근사의 한계

- reverse KL의 mode-seeking, 불확실성 과소평가

## 실습

- 같은 사후분포를 MCMC·변분추론·Laplace 근사로 구해 모양과 불확실성을 비교한다.

## 실패 사례

- 변분 사후의 분산을 실제 불확실성으로 받아들이는 것

## 통과 기준

- ELBO를 두 가지 방식(Jensen, KL 분해)으로 유도한다.
