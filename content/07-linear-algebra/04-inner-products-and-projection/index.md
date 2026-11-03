---
id: linear-algebra/inner-products-and-projection
title: "내적·노름·직교성·사영"
title_en: "Inner Products, Norms, and Projection"
tier: core
status: outline
prerequisites: [linear-algebra/rank-and-subspaces]
---

# 내적·노름·직교성·사영

## 학습 목표

- 내적과 노름의 종류를 구분한다
- 사영과 최소제곱을 유도한다
- pseudoinverse를 사용한다

## 내적과 노름

- 내적의 공리, 벡터 노름(L1 · L2 · L∞), 행렬 노름(Frobenius · spectral · nuclear)
- 거리와 유사도: cosine similarity의 기하

## 직교성

- 직교 · 정규직교 basis, Gram–Schmidt, 직교행렬의 성질

## 사영

- 부분공간으로의 사영, 사영행렬

## 최소제곱

- 정규방정식, 기하적 해석, 잔차의 직교성

## pseudoinverse

- 최소 노름 해, 미결정 시스템

## 실습

- 최소제곱 해를 정규방정식·QR·pseudoinverse로 각각 구하고 결과와 수치 안정성을 비교한다.

## 실패 사례

- 정규방정식을 그대로 풀어 조건수를 제곱하는 것
- 스케일이 다른 특징에 유클리드 거리를 쓰는 것

## 통과 기준

- 선형회귀를 사영으로 설명하고 잔차가 column space에 직교함을 보인다.
