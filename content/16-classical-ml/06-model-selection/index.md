---
id: classical-ml/model-selection
title: "모델 선택과 평가"
title_en: "Model Selection and Evaluation"
tier: core
status: outline
prerequisites: [classical-ml/baselines, statistical-inference/comparing-models]
---

# 모델 선택과 평가

## 학습 목표

- 교차검증을 문제 구조에 맞게 사용한다
- 적절한 metric을 고른다
- 모델 선택과 최종 보고를 분리한다

## 교차검증

- k-fold · group k-fold · time-series CV, nested CV의 필요성

## 회귀 metric

- MAE · RMSE · MAPE · R² · 상대오차, 단위와 해석

## 분류 metric

- accuracy · precision · recall · F1 · ROC-AUC · PR-AUC, class imbalance

## threshold 선택

- 비용 기반 threshold, calibration과의 관계

## 하이퍼파라미터 탐색

- grid · random · 베이지안 최적화, 탐색 예산의 공정성

## 진단

- 학습 곡선 · 검증 곡선 · 잔차 분석 · 조건별 오차

## 실습

- 같은 모델을 nested CV와 단순 CV로 평가해 성능 추정의 낙관 편향을 측정한다.

## 실패 사례

- 불균형 데이터에 accuracy를 쓰는 것
- 모델 선택에 쓴 점수를 최종 성능으로 보고하는 것

## 통과 기준

- 모델 선택에 사용한 데이터와 최종 성능 보고에 사용한 데이터를 명확히 구별한 평가 보고서를 쓴다.
