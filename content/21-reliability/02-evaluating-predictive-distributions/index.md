---
id: reliability/evaluating-predictive-distributions
title: "예측분포와 구간의 평가"
title_en: "Evaluating Predictive Distributions"
tier: core
status: outline
prerequisites: [reliability/uncertainty-quantification]
---

# 예측분포와 구간의 평가

## 학습 목표

- 적정 점수 규칙으로 예측분포를 평가한다
- calibration과 sharpness를 측정한다

## 적정 점수 규칙

- NLL · CRPS · Brier score · interval score

## calibration

- reliability diagram · ECE · PIT histogram, 회귀의 calibration

## sharpness

- 좁은 구간과 정확한 구간의 trade-off

## 구간 평가

- coverage와 폭, 조건별·집단별 calibration

## 재보정

- temperature scaling · isotonic regression

## 실습

- 여러 불확실성 방법을 NLL·CRPS·coverage·폭·조건별 calibration으로 비교한다.

## 실패 사례

- 평균 coverage만 보고 조건별 실패를 놓치는 것

## 통과 기준

- 평균 정확도 외에 calibration과 조건별 coverage를 포함한 평가 표를 만든다.
