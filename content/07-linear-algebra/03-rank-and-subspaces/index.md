---
id: linear-algebra/rank-and-subspaces
title: "행렬의 네 부분공간과 rank"
title_en: "The Four Subspaces and Rank"
tier: core
status: outline
prerequisites: [linear-algebra/vector-spaces]
---

# 행렬의 네 부분공간과 rank

## 학습 목표

- column·row·null·left null space를 구한다
- rank–nullity로 해의 존재와 유일성을 판단한다
- 딥러닝에서의 저랭크 구조를 이해한다

## 네 부분공간

- column space · row space · null space · left null space와 직교 관계

## rank

- rank의 여러 정의, 곱의 rank, full rank의 의미

## rank–nullity와 선형시스템

- 해의 존재 · 유일성 · 무한히 많은 해, 과결정·미결정 시스템

## 수치적 rank

- 정확한 rank와 수치적 rank, 특이값과의 연결(이후 장)

## 저랭크 구조

- 저랭크 가중치·업데이트(LoRA 등), 저랭크 행렬로서의 데이터

## 실습

- rank가 부족한 회귀 문제를 만들고 해의 비유일성과 정규화가 해를 선택하는 방식을 확인한다.

## 실패 사례

- 수치적으로 거의 의존적인 열을 독립으로 취급하는 것
- rank와 행렬 크기를 혼동하는 것

## 통과 기준

- 임의의 행렬의 네 부분공간을 구하고 해당 선형시스템의 해 구조를 설명한다.
