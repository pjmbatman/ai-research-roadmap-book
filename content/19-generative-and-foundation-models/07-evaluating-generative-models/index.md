---
id: generative-and-foundation-models/evaluating-generative-models
title: "생성모델의 평가"
title_en: "Evaluating Generative Models"
tier: core
status: outline
prerequisites: [generative-and-foundation-models/generative-modeling]
---

# 생성모델의 평가

## 학습 목표

- 생성 결과의 품질·다양성·커버리지를 평가한다
- 하위 과제로 유용성을 검증한다

## 평가 축

- 품질 · 다양성 · 커버리지 · 기억(memorization)

## 지표

- likelihood · FID류 지표 · precision/recall · 분포 거리(MMD · Wasserstein)

## 도메인 지표

- 물리 법칙 만족 여부, 통계적 특성 일치

## 하위 과제 평가

- 생성 데이터로 학습한 모델의 실제 데이터 성능

## 실습

- 생성된 시계열을 분포 지표·물리 일관성·하위 과제 유용성으로 평가한다.

## 실패 사례

- 한 가지 지표로 생성모델을 평가하는 것

## 통과 기준

- 보기 좋은 결과와 분포를 잘 반영한 결과를 구분하는 평가 체계를 제시한다.
