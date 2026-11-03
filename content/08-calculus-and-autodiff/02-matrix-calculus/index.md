---
id: calculus-and-autodiff/matrix-calculus
title: "행렬 미분"
title_en: "Matrix Calculus"
tier: core
status: outline
prerequisites: [calculus-and-autodiff/multivariable-calculus]
---

# 행렬 미분

## 학습 목표

- 행렬·벡터에 대한 미분을 유도한다
- layout 규약을 일관되게 쓴다
- 신경망 gradient를 손으로 유도한다

## layout 규약

- numerator · denominator layout, 책 전체의 규약

## 미분형(differential) 접근

- dL = tr(G^T dX)로 gradient 읽기

## 자주 쓰는 항등식

- 선형 · 이차형식 · trace · log det · 역행렬의 미분

## 신경망의 gradient

- 선형회귀, softmax cross-entropy, 2층 MLP의 gradient 유도

## 실습

- 선형회귀와 2층 MLP의 gradient를 손으로 유도하고 NumPy로 구현한 뒤 수치미분으로 검증한다.

## 실패 사례

- 전치를 빠뜨려 shape가 맞지 않는 gradient
- 서로 다른 layout 규약의 공식을 섞는 것

## 통과 기준

- softmax cross-entropy의 gradient가 (p − y)임을 유도한다.
