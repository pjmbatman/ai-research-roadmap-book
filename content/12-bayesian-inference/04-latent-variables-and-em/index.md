---
id: bayesian-inference/latent-variables-and-em
title: "잠재변수 모델과 EM"
title_en: "Latent Variable Models and EM"
tier: core
status: outline
prerequisites: [bayesian-inference/bayesian-thinking]
---

# 잠재변수 모델과 EM

## 학습 목표

- 잠재변수와 주변화를 이해한다
- EM 알고리즘을 유도하고 구현한다

## 잠재변수 모델

- 관측되지 않는 원인, 주변 우도

## Gaussian mixture model

- 혼합 성분, 책임도(responsibility)

## EM 알고리즘

- E-step · M-step, ELBO 관점의 유도, 단조 증가

## 결측 데이터

- 결측을 잠재변수로 다루기, 결측 메커니즘(MCAR · MAR · MNAR)

## 실습

- GMM을 EM으로 직접 구현하고 초기값에 따른 국소해 문제를 확인한다.

## 실패 사례

- EM이 전역 최적을 찾는다고 가정하는 것

## 통과 기준

- EM을 ELBO 최대화로 설명하고 VAE와의 관계를 말한다.
