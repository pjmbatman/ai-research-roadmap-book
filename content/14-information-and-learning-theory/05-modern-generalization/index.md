---
id: information-and-learning-theory/modern-generalization
title: "현대 딥러닝의 일반화와 스케일링"
title_en: "Generalization and Scaling in Modern Deep Learning"
tier: advanced
status: outline
prerequisites: [information-and-learning-theory/generalization-basics]
---

# 현대 딥러닝의 일반화와 스케일링

## 학습 목표

- 과매개변수화 모델의 일반화 현상을 안다
- 스케일링 법칙을 해석한다
- 보간과 외삽을 구분한다

## 과매개변수화

- interpolation regime, double descent

## 암묵적 정규화

- SGD · 초기화 · 구조가 만드는 편향

## 이론적 관점

- NTK와 lazy regime, feature learning, μP의 아이디어

## 스케일링 법칙

- 데이터 · 파라미터 · 계산량의 관계, compute-optimal 학습, 해석의 주의점

## 보간과 외삽

- 학습 분포 밖에서의 거동, 분포 이동(신뢰성 부와 연결)

## 실습

- 모델 크기와 데이터 크기를 바꾸어 작은 규모의 스케일링 곡선을 그리고 외삽 예측의 신뢰도를 평가한다.

## 실패 사례

- 작은 범위에서 얻은 스케일링 법칙을 멀리 외삽하는 것

## 통과 기준

- double descent와 스케일링 법칙을 자신의 실험 설계 결정에 연결한다.
