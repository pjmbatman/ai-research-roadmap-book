---
id: reliability/interpretability
title: "해석가능성"
title_en: "Interpretability"
tier: advanced
status: outline
prerequisites: [reliability/uncertainty-quantification]
---

# 해석가능성

## 학습 목표

- 주요 해석 방법과 가정을 이해한다
- 해석 결과를 검증한다

## 특징 기여도

- permutation importance · SHAP · gradient 기반 방법 · integrated gradients

## 방법의 가정

- 특징 독립 가정, 기준값 선택

## 설명과 인과

- attribution은 인과적 설명이 아니다

## 민감도 분석과 물리적 타당성

- 물리적으로 기대되는 반응과의 비교

## 기계적 해석

- 회로·특징 단위 해석의 아이디어

## 해석의 검증

- sanity check, 개입 실험

## 실습

- 같은 모델에 세 가지 attribution 방법을 적용해 결과가 어떻게 다른지 비교하고, 개입 실험으로 검증한다.

## 실패 사례

- 설명 그림 하나로 물리적 인과관계를 주장하는 것

## 통과 기준

- 해석 결과를 별도의 개입·검증 실험으로 뒷받침한다.
