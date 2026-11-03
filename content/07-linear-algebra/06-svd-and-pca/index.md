---
id: linear-algebra/svd-and-pca
title: "SVD·PCA·저랭크 근사"
title_en: "SVD, PCA, and Low-Rank Approximation"
tier: core
status: outline
prerequisites: [linear-algebra/eigen-and-symmetric]
---

# SVD·PCA·저랭크 근사

## 학습 목표

- SVD를 기하적으로 해석한다
- 저랭크 근사의 최적성을 이해한다
- PCA를 SVD로 구현한다

## SVD

- 회전–스케일–회전으로서의 SVD, 특이값과 네 부분공간

## 저랭크 근사

- Eckart–Young 정리, 에너지(분산) 비율

## PCA

- 분산 최대화와 재구성 오차 최소화, 중심화, whitening

## 계산

- 대용량에서의 randomized SVD, truncated SVD

## 활용

- 차원 축소, 노이즈 제거, 모델 압축, 표현 분석

## 실습

- 직접 구현한 SVD 기반 PCA와 라이브러리 결과를 비교하고, 차원에 따른 재구성 오차를 분석한다.

## 실패 사례

- 중심화 없이 PCA를 수행하는 것
- 단위가 다른 특징을 표준화 없이 PCA하는 것

## 통과 기준

- SVD로 PCA·저랭크 근사·pseudoinverse를 모두 설명한다.
