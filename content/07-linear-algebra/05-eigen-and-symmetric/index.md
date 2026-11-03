---
id: linear-algebra/eigen-and-symmetric
title: "고윳값과 대칭행렬"
title_en: "Eigenvalues and Symmetric Matrices"
tier: core
status: outline
prerequisites: [linear-algebra/inner-products-and-projection]
---

# 고윳값과 대칭행렬

## 학습 목표

- 고유분해와 대각화를 이해한다
- 대칭행렬의 스펙트럴 정리를 사용한다
- 양의 (준)정부호 행렬과 이차형식을 해석한다

## 고윳값과 고유벡터

- 특성방정식, 대각화 가능성, 반복 적용과 거듭제곱법

## 스펙트럴 정리

- 대칭행렬의 직교 대각화

## 양의 정부호성과 이차형식

- PSD · PD 판정, 이차형식의 기하(타원), Rayleigh quotient

## 공분산 행렬

- 공분산의 고유구조, 주축

## 동역학과 안정성

- 선형 시스템 x_{t+1}=Ax_t의 고윳값과 안정성, RNN gradient와의 연결

## 실습

- 거듭제곱법을 구현하고 선형 동역학 시스템의 장기 거동을 고윳값으로 예측한다.

## 실패 사례

- 비대칭 행렬에 대칭행렬의 성질을 적용하는 것
- 고윳값과 특이값을 혼동하는 것

## 통과 기준

- 공분산 행렬의 고유분해로 데이터의 주축을 설명하고, 선형 시스템의 안정성을 판정한다.
