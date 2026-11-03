---
id: optimization/nonconvex-landscape
title: "비볼록 최적화와 손실 지형"
title_en: "Nonconvex Optimization and Loss Landscapes"
tier: advanced
status: outline
prerequisites: [optimization/adaptive-methods]
---

# 비볼록 최적화와 손실 지형

## 학습 목표

- 딥러닝 손실 지형의 특성을 이해한다
- 손실 지형 분석 도구를 사용한다

## 비볼록성

- 국소해와 안장점, 고차원에서의 안장점

## 평탄함과 일반화

- sharp · flat minima 논쟁, SAM의 아이디어

## 학습 동역학

- 큰 학습률의 효과, edge of stability의 직관

## 손실 지형 분석

- 1차원·2차원 단면, mode connectivity, Hessian 스펙트럼

## 실습

- 학습된 두 모델 사이의 선형 보간 경로에서 손실을 측정하고 Hessian 최대 고윳값을 추정한다.

## 실패 사례

- 2차원 단면 시각화를 전체 지형으로 해석하는 것

## 통과 기준

- 학습 실패를 손실 지형과 학습 동역학 관점에서 설명한다.
