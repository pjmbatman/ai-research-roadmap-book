---
id: deep-learning-pytorch/debugging-training
title: "학습 실패 진단"
title_en: "Debugging Neural Network Training"
tier: core
status: outline
prerequisites: [deep-learning-pytorch/dashboards]
---

# 학습 실패 진단

## 학습 목표

- 학습 실패를 체계적으로 진단한다
- 디버깅 체크리스트를 사용한다

## 단계적 접근

- 데이터 확인 → 단순 모델 → batch 하나 과적합 → 규모 확대

## 증상별 진단

- 손실이 줄지 않음 · NaN · 발산 · 검증 성능 정체 · 이상하게 좋은 성능

## 흔한 원인

- 데이터·label 오류, 정규화 누락, 학습률, 초기화, mask, eval 모드, 손실 집계

## 도구

- gradient norm 추적, anomaly detection, 활성값 통계

## 디버깅 기록

- 가설 · 실험 · 결과를 남기는 습관

## 실습

- 일부러 버그 다섯 개를 심은 학습 코드에서 체크리스트로 모든 버그를 찾는다.

## 실패 사례

- 여러 설정을 한꺼번에 바꾸어 원인을 알 수 없게 되는 것

## 통과 기준

- 학습 실패 사례를 원인·근거·해결로 기록한 보고서를 쓴다.
