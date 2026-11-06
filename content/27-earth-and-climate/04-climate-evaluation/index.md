---
id: earth-and-climate/climate-evaluation
title: "기상·기후 예측의 평가"
title_en: "Evaluating Weather and Climate Predictions"
tier: core
status: outline
prerequisites: [earth-and-climate/weather-forecasting, reliability/evaluating-predictive-distributions]
---

# 기상·기후 예측의 평가

## 학습 목표

- 기상 분야의 표준 지표를 사용한다
- 흐릿한 예측의 함정을 피한다

## 결정론적 지표

- 면적 가중 RMSE, anomaly correlation coefficient

## 확률적 지표

- CRPS, spread–skill 관계, rank histogram

## 흐릿함의 함정

- 평균 손실이 만드는 평활화, 에너지 스펙트럼으로 진단

## 극한과 사건

- 극한 강수·폭염, 사건 기반 지표

## 물리적 일관성

- 균형 관계, 보존량

## 실습

- 같은 예보에 RMSE와 에너지 스펙트럼·CRPS를 함께 적용해 RMSE만으로는 보이지 않는 차이를 찾는다.

## 실패 사례

- RMSE만으로 흐릿한 예보를 더 좋다고 판단하는 것

## 통과 기준

- 결정론·확률·스펙트럼·극한 지표를 포함한 평가표를 만든다.
