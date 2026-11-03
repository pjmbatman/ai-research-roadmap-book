---
id: deep-learning-pytorch/tensors-and-autograd
title: "PyTorch 텐서와 autograd"
title_en: "PyTorch Tensors and Autograd"
tier: core
status: outline
prerequisites: [python-and-computing/scientific-python, calculus-and-autodiff/automatic-differentiation]
---

# PyTorch 텐서와 autograd

## 학습 목표

- 텐서의 shape·dtype·device를 다룬다
- autograd의 동작을 정확히 이해한다

## 텐서

- 생성 · dtype · device, NumPy와의 메모리 공유

## shape 연산

- indexing · broadcasting · view/reshape · transpose · permute · contiguous

## autograd

- requires_grad, 계산 그래프, backward, grad 누적

## 그래프 제어

- detach · no_grad · inference_mode · clone, in-place 연산의 위험

## 함수형 도구

- torch.func(vmap · grad · jacrev)의 개념

## 실습

- in-place 연산·detach 누락·grad 누적으로 생기는 버그를 재현하고 수정한다.

## 실패 사례

- detach 없이 손실 기록용 텐서를 쌓아 메모리가 새는 것
- view 이후 in-place 수정으로 gradient가 깨지는 것

## 통과 기준

- 임의의 코드에서 어떤 텐서가 그래프에 연결되는지 예측한다.
