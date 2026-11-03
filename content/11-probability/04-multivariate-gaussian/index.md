---
id: probability/multivariate-gaussian
title: "다변량 가우시안"
title_en: "The Multivariate Gaussian"
tier: core
status: outline
prerequisites: [probability/conditional-probability, linear-algebra/eigen-and-symmetric]
---

# 다변량 가우시안

## 학습 목표

- 다변량 가우시안의 주변·조건부 분포를 유도한다
- 선형-가우시안 모델을 다룬다

## 정의와 기하

- 평균과 공분산, 타원 등고선, Mahalanobis 거리

## 주변화와 조건부

- 블록 행렬 공식, Schur complement

## 선형 변환과 선형-가우시안 모델

- y=Ax+ε의 결합·주변·사후분포

## 정밀도 행렬

- 정밀도 행렬의 0과 조건부 독립

## 활용

- Gaussian process, Kalman filter, 베이지안 선형회귀, VAE의 공통 기반

## 실습

- 2차원 가우시안에서 조건부분포를 공식과 샘플링으로 각각 구해 비교한다.

## 실패 사례

- 공분산 행렬이 PSD가 아니게 되는 수치 문제

## 통과 기준

- 선형-가우시안 모델의 사후분포를 직접 유도한다.
