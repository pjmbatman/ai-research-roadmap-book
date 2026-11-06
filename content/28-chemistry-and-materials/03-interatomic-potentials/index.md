---
id: chemistry-and-materials/interatomic-potentials
title: "기계학습 원자간 퍼텐셜"
title_en: "Machine-Learned Interatomic Potentials"
tier: advanced
status: outline
prerequisites: [chemistry-and-materials/molecular-representations, scientific-ml/geometric-models]
---

# 기계학습 원자간 퍼텐셜

## 학습 목표

- 에너지·힘을 학습하는 퍼텐셜을 이해한다
- 분자동역학에서의 안정성과 외삽을 평가한다

## 문제 정의

- 원자 배치 → 에너지, 힘은 에너지의 gradient

## 학습 데이터

- 양자역학 계산 데이터, 계산 수준의 일관성, 능동 학습으로 데이터 수집

## 모델

- 국소 환경 기술자 기반 모델, 등변 message passing

## 시뮬레이션에서의 평가

- 에너지·힘 오차, 분자동역학 안정성, 관측량 재현

## 실습

- 작은 분자계의 퍼텐셜을 학습해 힘 오차와 짧은 분자동역학의 안정성·에너지 보존을 함께 평가한다.

## 실패 사례

- 테스트셋 힘 오차가 작으면 분자동역학도 안정하다고 가정하는 것

## 통과 기준

- 퍼텐셜을 정적 오차와 동역학 안정성으로 함께 보고한다.
