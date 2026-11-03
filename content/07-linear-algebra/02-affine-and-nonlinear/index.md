---
id: linear-algebra/affine-and-nonlinear
title: "아핀 구조와 비선형성"
title_en: "Affine Structure and Nonlinearity"
tier: core
status: outline
prerequisites: [linear-algebra/vector-spaces]
---

# 아핀 구조와 비선형성

## 학습 목표

- 선형과 아핀의 차이를 설명한다
- 선형층을 쌓아도 선형인 이유를 증명한다
- 비선형 활성화가 표현력을 바꾸는 방식을 설명한다

## 선형 변환과 아핀 변환

- Ax와 Ax+b의 차이, 원점 보존, 동차좌표(homogeneous coordinates)로 아핀을 선형으로

## 결합의 종류

- 선형 결합 · 아핀 결합 · convex 결합, 아핀 집합과 convex 집합

## 선형층의 합성

- 아핀 변환의 합성은 아핀이다: 깊이가 표현력을 늘리지 않는 경우

## 비선형성의 역할

- 활성화 함수가 만드는 영역 분할, ReLU 신경망은 조각별 아핀(piecewise affine) 함수
- 보편 근사 정리의 직관과 한계

## 선형 근사로서의 미분

- 비선형 함수의 국소 아핀 근사(다음 장들과의 연결)

## 실습

- 활성화 없는 MLP와 ReLU MLP가 2차원 평면을 어떻게 변형하는지 시각화하고, ReLU 신경망의 선형 영역 수를 센다.

## 실패 사례

- bias가 있는 층을 '선형층'이라 부르며 성질을 혼동하는 것
- 깊이만 늘리면 표현력이 늘어난다고 생각하는 것

## 통과 기준

- Ax와 Ax+b의 수학적 차이, 선형층 합성의 결과, 비선형 활성화의 역할을 증명과 그림으로 설명한다.
