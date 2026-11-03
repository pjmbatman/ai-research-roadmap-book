---
id: reliability/conformal-prediction
title: "Conformal Prediction"
title_en: "Conformal Prediction"
tier: advanced
status: outline
prerequisites: [reliability/evaluating-predictive-distributions]
---

# Conformal Prediction

## 학습 목표

- conformal prediction의 보장과 가정을 이해한다
- split conformal을 구현한다

## 기본 아이디어

- 교환가능성, 비순응 점수, 분포 무관 coverage 보장

## split conformal

- 보정 데이터, 분위수 계산

## 변형

- CQR · 적응적 구간 · 조건부 coverage의 한계

## 시계열과 분포 이동

- 교환가능성이 깨질 때, 적응적 conformal

## 실습

- split conformal과 CQR을 구현하고 분포 이동 상황에서 coverage가 어떻게 무너지는지 확인한다.

## 실패 사례

- 주변 coverage 보장을 조건부 보장으로 오해하는 것

## 통과 기준

- conformal 보장이 성립하는 조건을 명시하고 실험으로 확인한다.
