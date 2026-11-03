---
id: reinforcement-learning/model-free-rl
title: "가치 기반·정책 기반 강화학습"
title_en: "Value-Based and Policy-Based RL"
tier: advanced
status: outline
prerequisites: [reinforcement-learning/mdp-and-dp]
---

# 가치 기반·정책 기반 강화학습

## 학습 목표

- 주요 model-free 알고리즘을 이해한다
- RL 실험의 재현성 문제를 안다

## 가치 기반

- TD 학습 · Q-learning · DQN

## 정책 기반

- policy gradient · REINFORCE · baseline

## actor-critic

- A2C · PPO · SAC

## 실험의 어려움

- 높은 분산, seed 민감도, 평가 프로토콜

## 실습

- PPO로 연속 제어 과제를 학습하고 seed 10개의 결과 분포를 보고한다.

## 실패 사례

- 단일 seed의 학습 곡선으로 알고리즘을 비교하는 것

## 통과 기준

- RL 실험을 여러 seed와 신뢰구간으로 보고한다.
