---
id: classical-ml/trees-and-ensembles
title: "트리와 앙상블"
title_en: "Trees and Ensembles"
tier: core
status: outline
prerequisites: [classical-ml/baselines]
---

# 트리와 앙상블

## 학습 목표

- 결정 트리·랜덤 포레스트·gradient boosting을 사용한다
- 표 데이터에서 강한 baseline을 만든다

## 결정 트리

- 분할 기준, 깊이와 과적합

## bagging과 랜덤 포레스트

- 분산 감소, OOB 오차

## gradient boosting

- 함수 공간에서의 gradient descent, XGBoost · LightGBM · CatBoost

## 표 데이터에서의 위치

- 딥러닝 대비 강점, 범주형 변수 처리

## 특징 중요도

- impurity 기반 중요도의 편향, permutation importance

## 실습

- 표 형식 공학 데이터에서 gradient boosting을 튜닝하고 신경망과 공정하게 비교한다.

## 실패 사례

- impurity 기반 중요도를 인과적 중요도로 해석하는 것

## 통과 기준

- 표 데이터 문제에서 잘 튜닝된 boosting baseline을 제시한다.
