---
id: git-and-agents/agentic-workflow
title: "에이전트 기반 연구 워크플로"
title_en: "Agentic Research Workflows"
tier: advanced
status: outline
prerequisites: [git-and-agents/mcp-and-tools]
---

# 에이전트 기반 연구 워크플로

## 학습 목표

- 계획-구현-검증-리뷰 흐름으로 에이전트를 운용한다
- 격리된 실험 환경을 사용한다
- 실행 위치와 승인 경계를 관리한다

## 작업 흐름

- 문제 정의 → 계획 → 구현 → 검증 → 리뷰, 사람이 승인하는 지점

## 격리

- branch · worktree · 컨테이너로 병렬 실험 분리

## 실행 위치

- 로컬·원격 서버·클라우드 에이전트, 다른 기기에서 세션 원격 제어

## 권한 경계

- 코드 변경 승인, 실험 실행, 외부 작업(push · 배포 · 메시지) 구분

## 연구 무결성

- 데이터·결과 조작 방지, benchmark·test set 누수 방지, 에이전트 작업 기록

## 실습

- 에이전트에게 작은 ablation 실험을 계획·구현·실행하게 하고, 결과와 변경을 직접 검증한 리뷰 기록을 남긴다.

## 실패 사례

- 에이전트가 test set을 보고 튜닝하게 두는 것
- 검토하지 않은 대량 변경을 merge하는 것

## 통과 기준

- 에이전트가 수행한 실험에서 연구 결론에 영향을 주는 가정을 직접 검증한다.
