---
id: architectures/language-models
title: "언어모델과 LLM의 기초"
title_en: "Language Models and LLM Foundations"
tier: advanced
status: outline
prerequisites: [architectures/transformers]
---

# 언어모델과 LLM의 기초

## 학습 목표

- 언어모델의 학습 과정을 이해한다
- LLM을 연구 도구와 연구 대상으로 다룬다

## 토큰화

- BPE · SentencePiece, 수치·과학 데이터의 토큰화 문제

## 자기회귀 언어모델

- next-token prediction, perplexity

## 사전학습과 후속 학습

- instruction tuning · RLHF · DPO의 개념

## 사용 방식

- prompting · in-context learning · RAG · tool use

## 평가

- 벤치마크와 오염(contamination), LLM 평가의 어려움

## 과학 연구에서의 LLM

- 문헌 · 코드 · 데이터 해석 보조, 한계와 검증

## 실습

- 작은 문자 단위 언어모델을 학습하고, 사전학습된 LLM으로 도메인 문서 RAG를 구성해 답변을 검증한다.

## 실패 사례

- LLM 출력을 사실 확인 없이 연구 근거로 쓰는 것

## 통과 기준

- LLM의 학습 단계와 사용 방식을 구분해 설명한다.
