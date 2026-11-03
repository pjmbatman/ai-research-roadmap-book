---
id: deep-learning-pytorch/datasets-and-dataloaders
title: "Dataset과 DataLoader"
title_en: "Datasets and DataLoaders"
tier: core
status: outline
prerequisites: [deep-learning-pytorch/tensors-and-autograd, data/splitting-and-leakage]
---

# Dataset과 DataLoader

## 학습 목표

- 데이터 계약(sample contract)을 설계한다
- 가변 길이 데이터를 batch로 만든다
- DataLoader의 병렬 동작을 이해한다

## Dataset

- map-style과 iterable-style, __getitem__의 반환 계약

## sample contract

- 입력 · 정답 · mask · metadata를 담은 구조, shape와 단위 명시

## batch 구성

- collate_fn, padding과 mask, 가변 길이 시계열

## sampler

- shuffle, weighted sampler, group 단위 sampler

## 병렬 로딩

- num_workers · pin_memory · prefetch · persistent workers, worker별 seed

## 대용량 데이터

- streaming, memory mapping, 캐싱

## 실습

- 가변 길이 시계열을 padding·mask로 batch화하는 Dataset·collate_fn을 작성하고 테스트한다.

## 실패 사례

- worker들이 같은 random seed를 공유하는 것
- padding된 값이 손실과 metric에 섞이는 것

## 통과 기준

- 데이터 로더를 단독으로 테스트하고 batch 하나를 사람이 검증할 수 있게 시각화한다.
