---
id: deep-learning-pytorch/pytorch-ecosystem
title: "PyTorch 생태계와 대안"
title_en: "The PyTorch Ecosystem and Alternatives"
tier: reference
status: outline
prerequisites: [deep-learning-pytorch/trainer-design]
---

# PyTorch 생태계와 대안

## 학습 목표

- 주요 라이브러리의 역할을 안다
- JAX 등 대안 프레임워크의 특징을 이해한다

## 컴파일과 성능

- torch.compile의 개념과 주의점

## 도메인 라이브러리

- Hugging Face(transformers · datasets) · timm · PyG · torchvision · torchaudio

## 과학 계산 연계

- torchdiffeq · 미분가능 시뮬레이터

## 대안 프레임워크

- JAX(함수형 변환 · jit · vmap) · Flax · Equinox, 선택 기준

## 라이브러리 선택 원칙

- 유지보수 상태 · 라이선스 · 의존성 무게

## 실습

- 같은 작은 모델을 PyTorch와 JAX로 구현해 코드 구조와 성능을 비교한다.

## 실패 사례

- 유지보수가 끝난 라이브러리에 연구 전체를 의존하는 것

## 통과 기준

- 연구에 필요한 라이브러리를 근거와 함께 선택한다.
