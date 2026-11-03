---
id: classical-ml/baselines
title: "기준 모델의 철학"
title_en: "The Philosophy of Baselines"
tier: core
status: outline
prerequisites: [data/splitting-and-leakage]
---

# 기준 모델의 철학

## 학습 목표

- 문제에 맞는 단순 baseline을 설계한다
- baseline이 연구 주장에 주는 의미를 이해한다

## 왜 baseline인가

- 개선의 기준점, 문제 난이도의 측정

## 단순 baseline

- 평균 · 최근값(persistence) · 선형 추세 · 물리식 · 룩업 테이블

## 강한 baseline

- 잘 튜닝된 고전 모델, 공정한 튜닝 예산

## baseline이 이기는 경우

- 문제 정의·데이터·평가를 다시 보라는 신호

## 실습

- 딥러닝 전에 동일한 분할·metric으로 단순 baseline 다섯 개를 비교한다.

## 실패 사례

- 약한 baseline만 골라 비교하는 것
- baseline을 튜닝하지 않는 것

## 통과 기준

- 연구 문제의 baseline 표를 만들고 각 baseline의 의미를 설명한다.
