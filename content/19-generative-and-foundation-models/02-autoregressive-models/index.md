---
id: generative-and-foundation-models/autoregressive-models
title: "자기회귀 모델"
title_en: "Autoregressive Models"
tier: core
status: outline
prerequisites: [generative-and-foundation-models/generative-modeling]
---

# 자기회귀 모델

## 학습 목표

- chain rule 분해로 분포를 모델링한다
- 샘플링 전략을 이해한다

## chain rule 분해

- 순서의 선택, 정확한 likelihood

## 구조

- masked convolution · causal Transformer

## 샘플링

- greedy · temperature · top-k · nucleus

## 연속값의 자기회귀

- 분포 head, 이산화

## 실습

- 연속 시계열을 위한 자기회귀 모델을 만들고 샘플링 온도에 따른 다양성을 평가한다.

## 실패 사례

- 학습과 생성의 조건 차이(exposure bias)를 무시하는 것

## 통과 기준

- 자기회귀 모델의 likelihood와 샘플링을 구현한다.
