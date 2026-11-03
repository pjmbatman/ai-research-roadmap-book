---
id: environments/software-stack
title: "AI 소프트웨어 스택의 계층"
title_en: "Layers of the AI Software Stack"
tier: core
status: outline
prerequisites: [linux-and-remote/linux-shell]
---

# AI 소프트웨어 스택의 계층

## 학습 목표

- 하드웨어부터 프레임워크까지의 계층을 설명한다
- CUDA 관련 버전 정보를 구분해 읽는다

## 계층 구조

- 하드웨어(CPU 아키텍처 · GPU) → OS·kernel → NVIDIA driver → CUDA runtime·toolkit → cuDNN·NCCL → Python → PyTorch

## 버전 정보 읽기

- nvidia-smi의 CUDA Version, nvcc --version, torch.version.cuda가 각각 의미하는 것
- driver의 하위 호환성과 forward compatibility

## PyTorch 바이너리와 CUDA

- wheel에 포함된 CUDA runtime, 별도 toolkit이 필요한 경우(custom extension, 컴파일)

## 컴파일러와 ABI

- gcc 버전, C++ ABI, custom CUDA extension 빌드 실패의 원인

## 실습

- 자신의 서버에서 각 계층의 버전을 수집하는 스크립트를 작성하고 결과를 해석한다.

## 실패 사례

- nvidia-smi의 CUDA Version을 설치된 toolkit 버전으로 오해하는 것
- PyTorch 사용을 위해 toolkit을 무조건 설치하는 것

## 통과 기준

- 임의의 서버에서 계층별 버전을 수집하고 호환성 여부를 판단한다.
