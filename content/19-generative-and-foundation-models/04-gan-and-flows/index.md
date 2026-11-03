---
id: generative-and-foundation-models/gan-and-flows
title: "GAN과 Normalizing Flow"
title_en: "GANs and Normalizing Flows"
tier: advanced
status: outline
prerequisites: [generative-and-foundation-models/generative-modeling, calculus-and-autodiff/integration]
---

# GAN과 Normalizing Flow

## 학습 목표

- 적대적 학습의 원리와 불안정성을 이해한다
- 가역 변환으로 정확한 likelihood를 계산한다

## GAN

- 생성자와 판별자, minimax 목적, divergence 해석

## GAN의 어려움

- mode collapse, 학습 불안정, 안정화 기법

## Normalizing flow

- 변수변환 공식, 가역성과 Jacobian 행렬식

## flow 구조

- coupling layer · autoregressive flow · 연속 flow

## 실습

- 같은 데이터에 GAN과 flow를 학습해 샘플 품질과 likelihood 계산 가능성을 비교한다.

## 실패 사례

- GAN의 손실 값으로 학습 진행을 판단하는 것

## 통과 기준

- flow의 log-likelihood 계산을 변수변환 공식으로 설명한다.
