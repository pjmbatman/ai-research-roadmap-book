---
id: linear-algebra/tensors-and-einsum
title: "텐서와 einsum"
title_en: "Tensors and einsum"
tier: core
status: outline
prerequisites: [linear-algebra/svd-and-pca]
---

# 텐서와 einsum

## 학습 목표

- 다차원 배열 연산을 인덱스 표기로 표현한다
- einsum으로 복잡한 연산을 작성한다
- shape 대수를 익힌다

## 다차원 배열로서의 텐서

- 축(axis)의 의미 부여: batch · time · channel · feature

## 인덱스 표기와 einsum

- 아인슈타인 표기법, 축약(contraction), batch 행렬곱

## 주요 연산

- outer product, Kronecker product, 텐서 축약, 전치와 reshape의 차이

## shape 대수

- broadcasting의 수학적 규칙, shape 추적 습관

## 텐서 분해 개요

- CP · Tucker 분해의 개념(심화)

## 실습

- attention·bilinear layer·batch 공분산을 einsum으로 구현하고 loop 구현과 결과를 비교한다.

## 실패 사례

- reshape와 transpose를 혼동해 데이터를 섞는 것
- 축의 의미를 잃어버리는 것

## 통과 기준

- 복잡한 텐서 연산을 einsum 한 줄과 shape 주석으로 표현한다.
