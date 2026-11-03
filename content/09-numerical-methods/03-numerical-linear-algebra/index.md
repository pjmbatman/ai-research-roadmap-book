---
id: numerical-methods/numerical-linear-algebra
title: "수치 선형대수"
title_en: "Numerical Linear Algebra"
tier: core
status: outline
prerequisites: [numerical-methods/conditioning-and-stability, linear-algebra/svd-and-pca]
---

# 수치 선형대수

## 학습 목표

- 선형시스템의 직접법과 반복법을 선택한다
- 희소 행렬을 다룬다

## 직접법

- LU · QR · Cholesky 분해와 사용처

## 반복법

- Jacobi · Gauss–Seidel · conjugate gradient · GMRES

## 희소 행렬

- 저장 형식(CSR · CSC), sparse solver

## 대규모 계산

- 행렬을 만들지 않는(matrix-free) 계산, 무작위 선형대수

## 실습

- 대규모 희소 선형시스템을 직접법과 CG로 풀어 시간·메모리·정확도를 비교한다.

## 실패 사례

- 역행렬을 명시적으로 계산하는 것

## 통과 기준

- 문제 구조에 맞는 선형 solver를 고르고 그 이유를 설명한다.
