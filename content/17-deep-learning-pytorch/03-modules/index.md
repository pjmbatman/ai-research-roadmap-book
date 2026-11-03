---
id: deep-learning-pytorch/modules
title: "nn.Module과 모델 구성"
title_en: "nn.Module and Model Composition"
tier: core
status: outline
prerequisites: [deep-learning-pytorch/tensors-and-autograd]
---

# nn.Module과 모델 구성

## 학습 목표

- Module로 모델을 구성한다
- parameter와 buffer를 구분한다
- state_dict를 다룬다

## nn.Module

- __init__과 forward, 하위 모듈 등록

## parameter와 buffer

- 학습되는 것과 저장되는 것, register_buffer

## 컨테이너

- Sequential · ModuleList · ModuleDict, 일반 list를 쓰면 생기는 문제

## state_dict

- 저장 · 로드 · strict 옵션, 이름 불일치 처리

## hook

- forward·backward hook으로 내부 관찰

## 설정에서 모델 만들기

- config 기반 모델 생성, registry 패턴

## 실습

- 설정 파일로 구성되는 모델을 만들고, hook으로 중간 활성값을 기록한다.

## 실패 사례

- 일반 list에 넣은 층이 optimizer에 등록되지 않는 것

## 통과 기준

- 모델의 모든 파라미터·버퍼를 나열하고 state_dict로 정확히 복원한다.
