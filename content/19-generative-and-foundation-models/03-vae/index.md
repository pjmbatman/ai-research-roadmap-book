---
id: generative-and-foundation-models/vae
title: "VAE와 잠재변수 생성모델"
title_en: "Variational Autoencoders"
tier: core
status: outline
prerequisites: [generative-and-foundation-models/generative-modeling, bayesian-inference/variational-inference]
---

# VAE와 잠재변수 생성모델

## 학습 목표

- VAE를 ELBO로 유도한다
- 재매개변수화를 구현한다
- posterior collapse를 진단한다

## 모델

- decoder p(x|z) · prior p(z) · encoder q(z|x)

## ELBO

- 재구성 항과 KL 항, β-VAE

## 재매개변수화

- 가우시안 encoder의 gradient

## 문제와 대응

- posterior collapse, 흐릿한 샘플의 원인

## 확장

- 조건부 VAE, 계층적 VAE, VQ-VAE

## 실습

- VAE를 구현하고 β 값에 따른 재구성과 잠재 공간 구조를 비교한다.

## 실패 사례

- KL 항 가중치를 바꾸고도 ELBO라 부르는 것

## 통과 기준

- VAE의 목적함수를 유도하고 구현에서 각 항을 확인한다.
