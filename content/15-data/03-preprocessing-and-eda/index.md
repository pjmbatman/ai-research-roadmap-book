---
id: data/preprocessing-and-eda
title: "전처리와 탐색적 분석"
title_en: "Preprocessing and Exploratory Analysis"
tier: core
status: outline
prerequisites: [data/data-semantics]
---

# 전처리와 탐색적 분석

## 학습 목표

- 결측·이상치·센서 오류를 처리한다
- 정규화와 재표본화의 영향을 이해한다
- 시각화로 품질을 점검한다

## 품질 문제

- 결측값 · 이상치 · 중복 · 센서 오류 · 단위 오류

## 정규화와 스케일링

- standardization · min-max · robust scaling, 통계량을 어디서 추정하는가

## 시계열 전처리

- timestamp 정렬 · resampling · 보간 · filtering · smoothing

## 특징 추출

- 도메인 지식 기반 특징, 자동 특징

## 탐색적 분석과 시각화

- 분포 · 상관 · 시계열 개요 · 그룹별 비교, 대표 샘플 직접 보기

## 실습

- 전처리 전후의 분포와 대표 샘플을 비교하고, 보간·필터링으로 손실되는 정보를 기록한다.

## 실패 사례

- 데이터를 직접 보지 않고 바로 학습하는 것
- 전체 데이터로 정규화 통계량을 계산하는 것

## 통과 기준

- 전처리 각 단계의 근거와 영향을 시각화와 함께 문서화한다.
