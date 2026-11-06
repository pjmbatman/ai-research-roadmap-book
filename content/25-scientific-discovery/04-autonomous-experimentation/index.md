---
id: scientific-discovery/autonomous-experimentation
title: "자율 실험과 closed-loop 연구"
title_en: "Autonomous Experimentation and Closed-Loop Research"
tier: advanced
status: outline
prerequisites: [scientific-discovery/experimental-design]
---

# 자율 실험과 closed-loop 연구

## 학습 목표

- 모델·실험 선택·실험 수행이 이어지는 closed-loop 구조를 설계한다
- 자율 실험의 실패 양상을 안다

## closed-loop의 구성

- 가설·모델 → 실험 선택 → 실행(로봇·시뮬레이터) → 측정 → 모델 갱신

## 자율 실험실

- self-driving lab의 개념, 자동화 수준

## 다목적과 제약

- 성능·비용·안전 제약 아래의 선택

## 운영 문제

- 측정 실패, 장비 드리프트, 사람의 개입 지점, 기록과 재현

## 실습

- 시뮬레이터를 '가상 실험 장비'로 두고 closed-loop 최적화를 구성해, 측정 잡음과 실패를 넣었을 때의 거동을 관찰한다.

## 실패 사례

- 측정 실패를 정상 데이터로 학습하는 것
- 루프의 결정 기록을 남기지 않는 것

## 통과 기준

- closed-loop 실행의 모든 결정을 재현할 수 있는 기록과 함께 결과를 보고한다.
