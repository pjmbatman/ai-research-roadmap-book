---
id: architectures/transformers
title: "Attention과 Transformer"
title_en: "Attention and Transformers"
tier: core
status: outline
prerequisites: [architectures/inductive-bias, linear-algebra/tensors-and-einsum]
---

# Attention과 Transformer

## 학습 목표

- attention을 직접 구현한다
- mask와 위치 인코딩을 다룬다
- 계산 복잡도를 이해한다

## attention

- query · key · value, scaled dot-product, softmax의 역할

## multi-head와 cross-attention

- self · cross attention, head의 의미

## 위치 정보

- absolute · relative · RoPE, 순서 불변성 깨기

## mask

- causal mask · padding mask

## Transformer 블록

- encoder · decoder, pre-norm과 post-norm, FFN

## 효율

- 길이에 대한 이차 복잡도, efficient attention의 개념, KV cache

## 해석의 한계

- attention map은 설명이 아니다

## 실습

- 작은 attention 블록을 직접 구현하고 shape·mask·gradient를 테스트로 검증한다.

## 실패 사례

- padding mask 누락
- attention 가중치를 중요도로 해석하는 것

## 통과 기준

- 직접 구현한 Transformer가 PyTorch 구현과 같은 출력을 낸다.
