---
id: numerical-methods/conditioning-and-stability
title: "조건수·안정성·수렴"
title_en: "Conditioning, Stability, and Convergence"
tier: core
status: outline
prerequisites: [numerical-methods/floating-point]
---

# 조건수·안정성·수렴

## 학습 목표

- 문제의 조건수와 알고리즘의 안정성을 구분한다
- 수렴 속도를 분석한다

## 조건수

- 입력 섭동에 대한 출력 민감도, 행렬의 조건수

## 알고리즘의 안정성

- forward · backward stability

## 수렴

- 수렴 차수(선형 · 2차), 정지 조건

## 스케일링과 전처리

- 변수 스케일 맞추기, preconditioning

## 실습

- 조건이 나쁜 문제에서 수학적으로 동등한 계산식이 다른 결과를 내는 사례를 만들고 원인을 분석한다.

## 실패 사례

- 나쁜 조건의 문제를 알고리즘 탓으로 돌리는 것

## 통과 기준

- 주어진 수치 문제의 오차가 조건수 때문인지 알고리즘 때문인지 판별한다.
