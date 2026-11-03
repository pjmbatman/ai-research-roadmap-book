---
id: deep-learning-pytorch/training-loop
title: "학습 루프 해부: forward → loss → backward → update"
title_en: "Anatomy of a Training Loop"
tier: core
status: outline
prerequisites: [deep-learning-pytorch/modules, deep-learning-pytorch/datasets-and-dataloaders, optimization/adaptive-methods]
---

# 학습 루프 해부: forward → loss → backward → update

## 학습 목표

- end-to-end 학습 과정의 모든 단계를 설명하고 구현한다
- 수식과 코드를 대응시킨다

## 전체 흐름

- 원본 데이터 → sample → batch → device 이동 → forward → loss → gradient 초기화 · 누적 → backward → optimizer update → validation · logging · checkpoint

## 수식과 코드

- ŷ=f_θ(x), L=ℓ(ŷ,y), g=∇_θL, θ←Update(θ,g)

## 역할의 구분

- forward와 backward, gradient 계산과 parameter update, 학습 손실과 보고용 metric

## 단위의 구분

- batch · step · epoch, gradient accumulation 시 step의 의미

## 실전 요소

- mixed precision(autocast · GradScaler), gradient clipping, scheduler 호출 위치

## end-to-end라는 말의 의미

- 어떤 구성 요소가 실제로 공동 최적화되는가

## 실습

- 프레임워크 없이 학습 루프를 처음부터 작성하고, 각 줄을 수식에 대응시키는 주석을 단다.

## 실패 사례

- zero_grad 위치 오류로 gradient가 누적되는 것
- scheduler를 잘못된 주기로 호출하는 것

## 통과 기준

- 작은 데이터 하나에 과적합하는 데 성공하고, 학습 루프의 모든 줄을 설명한다.
