---
id: optimization/constrained-optimization
title: "제약 최적화와 쌍대성"
title_en: "Constrained Optimization and Duality"
tier: core
status: outline
prerequisites: [optimization/optimization-basics]
---

# 제약 최적화와 쌍대성

## 학습 목표

- Lagrangian과 KKT 조건을 사용한다
- 제약과 penalty·정규화의 관계를 이해한다

## Lagrange 승수

- 등식 제약, 기하적 해석

## KKT 조건

- 부등식 제약, 상보적 여유

## 쌍대성

- 쌍대 문제, 약·강 쌍대성

## 제약과 penalty

- penalty · barrier · augmented Lagrangian, 정규화 항과 노름 제약의 동치

## 투영과 근위 연산

- projected gradient, proximal operator(L1과 soft thresholding)

## 실습

- 같은 문제를 제약식과 penalty 방식으로 풀어 해와 수렴을 비교한다.

## 실패 사례

- penalty 계수를 키우기만 하면 제약이 만족된다고 생각하는 것

## 통과 기준

- Ridge 회귀를 노름 제약 문제로 다시 쓰고 KKT로 동치를 설명한다.
