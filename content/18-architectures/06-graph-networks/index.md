---
id: architectures/graph-networks
title: "그래프 신경망과 기하학적 딥러닝"
title_en: "Graph Neural Networks and Geometric Deep Learning"
tier: advanced
status: outline
prerequisites: [architectures/inductive-bias]
---

# 그래프 신경망과 기하학적 딥러닝

## 학습 목표

- message passing을 이해한다
- 대칭성을 구조에 반영한다

## 그래프 표현

- adjacency · node · edge feature

## message passing

- 집계 함수, GCN · GAT · GIN

## 대칭성

- permutation invariance · equivariance, E(n) 등변 네트워크의 개념

## 시공간 그래프

- 센서 네트워크 · 셀 간 연결

## 그래프 구성

- 물리적 연결과 학습된 연결, 그래프 구성 가정과 일반화

## 한계

- over-smoothing · over-squashing

## 실습

- 물리적 연결로 만든 그래프와 학습된 그래프로 같은 문제를 풀어 비교한다.

## 실패 사례

- 근거 없이 완전 연결 그래프를 쓰는 것

## 통과 기준

- 그래프 구성 방식을 데이터의 물리적 구조로 정당화한다.
