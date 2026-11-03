---
id: experiment-operations/distributed-training
title: "다중 GPU와 분산 학습"
title_en: "Multi-GPU and Distributed Training"
tier: advanced
status: outline
prerequisites: [experiment-operations/memory-management]
---

# 다중 GPU와 분산 학습

## 학습 목표

- 병렬화 방식을 구분한다
- DDP로 학습을 확장한다
- 분산 환경의 metric과 checkpoint를 다룬다

## 병렬화 방식

- data · tensor · pipeline parallelism, sharding(FSDP · ZeRO)

## DDP

- rank · world size · local rank, torchrun, collective 통신

## 분산 데이터

- DistributedSampler, epoch별 shuffle

## 학습 설정의 변화

- 유효 batch 크기 · 학습률 · step 수

## metric과 checkpoint

- 분산 집계, rank 0 저장, sharded checkpoint

## 통신 병목

- NCCL · interconnect, 디버깅 환경변수

## 실습

- 단일 GPU 학습을 DDP로 바꾸고, GPU 수에 따른 처리량과 학습 결과를 비교한다.

## 실패 사례

- 각 rank의 metric을 집계하지 않고 rank 0 값만 보고하는 것

## 통과 기준

- GPU 수가 늘 때 달라지는 batch·학습률·step 수·통신 비용을 설명한다.
