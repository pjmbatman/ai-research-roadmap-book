---
id: optimization/second-order-methods
title: "2차 방법과 준뉴턴"
title_en: "Second-Order and Quasi-Newton Methods"
tier: advanced
status: outline
prerequisites: [optimization/first-order-methods]
---

# 2차 방법과 준뉴턴

## 학습 목표

- Newton·Gauss–Newton·L-BFGS를 이해한다
- 파라미터 추정에 2차 방법을 사용한다

## Newton 방법

- 2차 근사, 수렴 속도, 비용

## Gauss–Newton과 Levenberg–Marquardt

- 비선형 최소제곱, 곡선 적합과 파라미터 추정

## 준뉴턴

- BFGS · L-BFGS

## 자연 gradient

- Fisher 정보 행렬, K-FAC의 개념

## 사용처

- 소규모 파라미터 추정, PINN 미세 최적화, 대규모 학습에서의 한계

## 실습

- 물리 모델 파라미터 추정을 Adam과 L-BFGS·LM으로 각각 풀어 수렴과 정확도를 비교한다.

## 실패 사례

- 확률적 mini-batch에 L-BFGS를 그대로 쓰는 것

## 통과 기준

- 문제 규모와 구조에 따라 1차·2차 방법을 선택한다.
