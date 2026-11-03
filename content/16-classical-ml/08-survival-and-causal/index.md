---
id: classical-ml/survival-and-causal
title: "생존분석과 인과추론 기초"
title_en: "Survival Analysis and Causal Inference Basics"
tier: advanced
status: outline
prerequisites: [classical-ml/time-series-analysis]
---

# 생존분석과 인과추론 기초

## 학습 목표

- censoring을 포함한 문제를 정의한다
- 상관과 인과를 구분한다
- 인과 그래프로 교란을 식별한다

## 생존분석

- censoring, survival function, hazard, Kaplan–Meier, Cox 모델

## 수명 예측과의 연결

- RUL을 회귀로 정의할 때와 censoring을 포함할 때의 차이

## 인과추론 기초

- 상관과 인과, confounding, 인과 DAG, backdoor 기준

## 개입과 반사실

- do 연산의 직관, 무작위 실험, 관찰 데이터의 한계

## 실습

- 실험이 중단된 셀을 포함한 수명 데이터를 단순 회귀와 생존분석으로 각각 모델링해 결과를 비교한다.

## 실패 사례

- 중단된 실험을 버리거나 수명이 끝난 것으로 취급하는 것
- 특징 중요도를 인과로 해석하는 것

## 통과 기준

- 자신의 연구 문제에서 censoring과 교란 요인을 식별하고 대응 방법을 제시한다.
