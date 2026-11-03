---
id: git-and-agents/github-collaboration
title: "GitHub 협업과 저장소 관리"
title_en: "Collaboration with GitHub"
tier: core
status: outline
prerequisites: [git-and-agents/git-fundamentals]
---

# GitHub 협업과 저장소 관리

## 학습 목표

- remote와 PR 기반 협업을 수행한다
- .gitignore와 대용량 파일 정책을 세운다
- 비밀정보 유출을 예방한다

## remote 작업

- clone · fetch · pull · push, upstream, fork

## Pull request와 code review

- PR 단위, 설명 작성, 리뷰 요청과 응답

## 커밋과 branch 규칙

- 커밋 메시지 규칙(Conventional Commits 등), branch 명명 규칙

## .gitignore

- 데이터 · 출력물 · 환경 · 캐시 · 비밀정보 제외, 이미 추적 중인 파일 제거

## 대용량 파일

- Git LFS의 범위와 한계, 데이터·checkpoint를 Git 밖에서 관리하는 이유

## 비밀정보

- API key · 토큰을 커밋했을 때의 대응, .env와 secret 관리, pre-commit 검사

## 이슈와 프로젝트 관리

- issue · label · milestone으로 연구 작업 추적

## 실습

- 두 사람이 같은 파일을 수정한 PR을 만들고 conflict를 해결해 merge한다.

## 실패 사례

- 데이터·checkpoint를 저장소에 커밋하는 것
- 비밀정보 커밋을 삭제 commit으로만 처리하는 것

## 통과 기준

- 연구 프로젝트 저장소를 .gitignore·PR 규칙과 함께 구성하고 협업 흐름을 시연한다.
