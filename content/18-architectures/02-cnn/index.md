---
id: architectures/cnn
title: "CNN과 공간 구조"
title_en: "Convolutional Networks"
tier: core
status: outline
prerequisites: [architectures/inductive-bias]
---

# CNN과 공간 구조

## 학습 목표

- convolution의 동작과 receptive field를 계산한다
- 1D·2D·3D CNN을 적용한다

## convolution

- kernel · stride · padding · dilation, 출력 크기 계산

## 귀납적 편향

- 국소성 · 가중치 공유 · 평행이동 등변성

## receptive field

- 깊이에 따른 증가, 유효 receptive field

## 대표 구조

- ResNet · encoder–decoder · U-Net

## 적용 범위

- 영상 · 신호 · 시계열에 대한 1D·2D CNN

## 실습

- 시계열 CNN의 receptive field가 실제 입력 구간을 얼마나 덮는지 계산하고 성능과 연결한다.

## 실패 사례

- receptive field가 필요한 맥락보다 짧은 모델

## 통과 기준

- CNN 설계를 receptive field와 데이터의 시간·공간 척도로 정당화한다.
