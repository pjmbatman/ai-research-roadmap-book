---
id: life-sciences/sequence-models
title: "생물 서열 모델"
title_en: "Biological Sequence Models"
tier: advanced
status: outline
prerequisites: [life-sciences/biology-for-ml, architectures/language-models]
---

# 생물 서열 모델

## 학습 목표

- 서열 모델과 사전학습 서열 모델을 사용한다
- 서열 표현의 전이 효과를 평가한다

## 서열의 표현

- one-hot, k-mer, 토큰화

## 서열 모델

- CNN · Transformer 기반 서열 모델, 긴 서열 처리

## 사전학습 언어모델

- 단백질·DNA 언어모델, embedding의 활용

## 진화 정보

- 다중 서열 정렬(MSA)과 공진화 신호

## 실습

- 사전학습 단백질 언어모델 embedding과 one-hot 표현으로 같은 기능 예측 과제를 풀어 비교한다.

## 실패 사례

- 유사 서열이 학습·테스트에 섞인 상태에서 사전학습 효과를 주장하는 것

## 통과 기준

- 서열 유사도 기반 분할로 사전학습 표현의 효과를 보인다.
