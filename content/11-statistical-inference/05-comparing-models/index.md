---
id: statistical-inference/comparing-models
title: "모델 비교의 통계"
title_en: "Statistics for Comparing Models"
tier: core
status: outline
prerequisites: [statistical-inference/intervals-and-testing]
---

# 모델 비교의 통계

## 학습 목표

- seed·데이터 분할에 따른 변동을 보고한다
- 짝지은 비교와 적절한 검정을 사용한다

## 변동의 출처

- seed · 데이터 분할 · 하이퍼파라미터 탐색 · 하드웨어 비결정성

## 짝지은 비교

- 같은 테스트 샘플에서의 차이, paired bootstrap

## 교차검증 결과의 비교

- fold 간 상관과 검정의 함정

## 여러 데이터셋에서의 비교

- 순위 기반 비교, critical difference

## 보고 방식

- 평균 · 표준편차 · 구간 · 실행 횟수, 표와 그림

## 실습

- 두 방법을 5개 seed와 그룹 분할로 비교하고, 차이를 주장할 수 있는지 통계적으로 판단한다.

## 실패 사례

- 단일 seed 결과로 우열을 주장하는 것
- 표준편차와 표준오차를 혼동하는 것

## 통과 기준

- 실험 결과 표를 변동성과 비교 방법까지 포함해 작성한다.
