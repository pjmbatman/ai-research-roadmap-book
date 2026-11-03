---
id: architectures/representation-analysis
title: "표현 분석"
title_en: "Analyzing Learned Representations"
tier: advanced
status: outline
prerequisites: [architectures/inductive-bias, classical-ml/unsupervised-learning]
---

# 표현 분석

## 학습 목표

- 학습된 표현을 정량적으로 분석한다
- 시각화의 한계를 안다

## probing

- linear probe, 무엇이 표현에 담겼는가

## 표현 유사도

- CKA · CCA, 층별·모델별 비교

## 검색과 최근접 이웃

- embedding 검색으로 표현 점검

## 시각화

- 저차원 투영의 해석 한계

## 실습

- 학습된 모델의 층별 표현에 linear probe를 적용해 물리량이 어디서 인코딩되는지 분석한다.

## 실패 사례

- probe의 높은 성능을 모델이 그 정보를 '사용한다'는 증거로 보는 것

## 통과 기준

- 표현 분석 결과를 대조 실험과 함께 보고한다.
