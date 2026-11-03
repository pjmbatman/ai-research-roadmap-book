---
id: classical-ml/time-series-analysis
title: "시계열 분석"
title_en: "Time Series Analysis"
tier: core
status: outline
prerequisites: [classical-ml/model-selection, probability/stochastic-processes]
---

# 시계열 분석

## 학습 목표

- 시계열의 구조를 분해한다
- 고전 예측 모델을 적용한다
- 시계열 평가 방식을 설계한다

## 시계열의 구조

- trend · seasonality · autocorrelation · 정상성

## 고전 모델

- AR · ARIMA · 지수평활, lag 특징

## 상태공간 모델

- 상태와 관측 방정식, Kalman filter와의 연결

## 예측·필터링·평활

- 문제 정의의 차이

## 평가

- rolling origin · backtesting, horizon별 오차

## 실습

- 시계열 예측 문제를 rolling-origin backtesting으로 평가하고 horizon별 오차를 분석한다.

## 실패 사례

- 시계열에 random split을 쓰는 것
- 미래 값을 사용한 특징을 만드는 것

## 통과 기준

- 시계열 문제의 평가 프로토콜을 설계하고 정당화한다.
