---
id: linear-algebra/function-spaces
title: "함수공간과 연산자"
title_en: "Function Spaces and Operators"
tier: advanced
status: outline
prerequisites: [linear-algebra/inner-products-and-projection]
---

# 함수공간과 연산자

## 학습 목표

- 함수를 벡터로 다루는 관점을 익힌다
- 연산자와 이산화의 관계를 이해한다
- 커널과 RKHS의 직관을 얻는다

## 함수를 벡터로

- 함수공간, L2 내적과 노름, 무한차원의 의미

## Hilbert 공간과 Banach 공간

- 완비성의 직관, 직교 basis 전개(Fourier · 다항식 basis)

## 연산자

- 함수에서 함수로의 사상, 미분·적분 연산자, 선형 연산자

## 이산화

- 연속 대상을 유한 차원으로 근사하기, 해상도와 이산화 오차

## 커널과 RKHS

- 양의 정부호 커널, 커널 트릭, 표현자 정리의 직관

## 실습

- 함수를 Fourier basis와 격자 값으로 각각 표현하고, 미분 연산자를 두 표현에서 행렬로 근사해 비교한다.

## 실패 사례

- 격자 위 값 벡터를 함수 자체로 착각해 해상도 변화에 무너지는 모델을 만드는 것

## 통과 기준

- neural operator와 Gaussian process가 함수공간 관점에서 무엇을 학습하는지 설명한다.
