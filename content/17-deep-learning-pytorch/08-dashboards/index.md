---
id: deep-learning-pytorch/dashboards
title: "학습 모니터링과 대시보드"
title_en: "Monitoring and Dashboards"
tier: core
status: outline
prerequisites: [deep-learning-pytorch/training-loop]
---

# 학습 모니터링과 대시보드

## 학습 목표

- 무엇을 기록할지 결정한다
- GUI 대시보드를 운영한다
- 학습 곡선을 해석한다

## 무엇을 기록하는가

- 손실 · metric · 학습률 · gradient norm · 파라미터 통계 · 처리량 · 예측 샘플

## TensorBoard

- scalar · histogram · image · text · hparams, 원격 접속

## 다른 도구

- Weights & Biases · MLflow · Aim, 로컬과 클라우드의 차이

## 학습 곡선 해석

- 과적합 · 과소적합 · 불안정 · 학습률 문제의 패턴

## 대시보드와 원본의 일치

- smoothing 착시, 로깅 주기, 원본 평가와의 대조

## 실습

- 원격 서버의 학습을 TensorBoard로 실시간 관찰하고, 이상 패턴을 하나 찾아 원인을 분석한다.

## 실패 사례

- smoothing된 곡선으로 결론을 내리는 것
- 대시보드 수치와 평가 스크립트 결과가 다른 것을 놓치는 것

## 통과 기준

- 학습 곡선만 보고 다음 실험의 조정 방향을 제안한다.
