---
id: scientific-ml/geometric-models
title: "대칭성을 반영한 과학 모델"
title_en: "Symmetry-Aware Models for Science"
tier: advanced
status: outline
prerequisites: [architectures/graph-networks, linear-algebra/symmetry-and-groups]
---

# 대칭성을 반영한 과학 모델

## 학습 목표

- 과학 데이터에 맞는 등변 모델을 선택한다
- 대칭 제약의 효과를 실험으로 검증한다

## 기하 데이터의 표현

- 점구름 · 그래프 · 격자 · 메시 · 구면 데이터

## 등변 신경망

- E(3) 등변 message passing, 스칼라·벡터 특징, 구면 조화함수 기반 표현의 개념

## 격자·메시·구면 위의 모델

- 등변 CNN, 메시 그래프 네트워크, 구면 위의 합성곱

## 대칭 제약의 효과

- 데이터 효율, 외삽, 계산 비용의 trade-off

## 대칭이 깨질 때

- 외부장·경계조건이 대칭을 깨는 경우의 처리

## 실습

- 같은 분자·입자 데이터에 등변 모델과 비등변 모델을 학습해 데이터 크기와 회전된 테스트셋에서의 성능을 비교한다.

## 실패 사례

- 대칭이 깨진 문제에 강한 등변 제약을 거는 것
- 회전된 테스트셋 없이 등변성의 효과를 주장하는 것

## 통과 기준

- 등변 제약의 효과를 데이터 효율과 회전 일반화 실험으로 보인다.
