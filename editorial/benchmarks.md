# UI/UX 벤치마킹

> 상태: **제안** (2026-11-06). 사용자 피드백 "디자인·UI/UX는 임의로 정하지 말고 인기 있는 레퍼런스를 벤치마킹한다"에 따라 작성했다.
> 원칙은 [design-system.md](design-system.md) §1, 화면 단위 규칙은 [ADR 0009](decisions/0009-paged-reading.md)를 따른다.

## 1. 조사 방법과 한계

- 세 갈래로 나누어 조사했다: ① 학습 플랫폼·로드맵 사이트, ② 문서·책·인터랙티브 튜토리얼 프레임워크, ③ 수학·ML 인터랙티브 설명(explorable explanation).
- 인기 근거는 GitHub 스타(제3자 집계 사이트, 집계 날짜 표기)·공시 사용자 수·수상 기록을 썼다. 조사 환경에서 github.com·API가 막혀 있어 스타 수는 **근삿값**이다.
- 레이아웃 수치(폭·비율·중단점)는 npm에 배포된 패키지 소스와 공개 저장소 원본 파일(raw)에서 직접 읽었다. 화면 관찰에만 근거한 항목은 "(관찰)"로 표시한다.

## 2. 레퍼런스

| 분류 | 레퍼런스 | 인기·평판 근거 | 우리가 배울 점 |
|---|---|---|---|
| 학습·로드맵 | [roadmap.sh](https://roadmap.sh) | 저장소 약 37만 스타, 등록 사용자 320만+ (2026-11 검색 기준) | 로드맵 그래프가 곧 진도 화면, 노드 클릭 시 옆 패널, 주제별 상태(대기·학습 중·완료·건너뜀)를 localStorage에 저장 |
| 학습 | [Brilliant](https://brilliant.org) | 학습자 약 1,000만 | 한 화면에 한 개념, 상호작용 문제마다 맞춤 피드백 |
| 학습 | [Duolingo](https://www.duolingo.com) | MAU 1.53억 (2026 Q3 SEC 공시) | 상단 진행 막대(숫자 없음), 하단 고정 '확인 → 피드백 → 계속' 버튼 (관찰) |
| 학습 | [freeCodeCamp](https://www.freecodecamp.org/learn) | 약 44–46만 스타 (2026) | 강의·워크숍·랩 3분류, 단계별 완료 표시가 안 보인다는 사용자 불만 → 단계 완료를 보여야 한다 |
| 학습 | [Exercism](https://exercism.org) | 트랙별 수만 명 | 지시문 \| 편집기, 테스트를 '명세'로 공개 |
| 학습 | [The Odin Project](https://www.theodinproject.com) | 커리큘럼 약 1.3만 스타 | 고정된 수업 형식, 확인 문제가 답이 있는 절로 링크 |
| 학습 | [Khan Academy](https://www.khanacademy.org) | 등록 1.9억 (2024–25 연차 보고서) | 3–4단계 숙련도 사다리 |
| 문서 프레임워크 | [Starlight](https://starlight.astro.build) | 약 9천 스타 (2026-07) | 본문 45rem, 줄간격 1.75, 13–24px 6단계 글자 크기, 콜아웃 4종, 탭 선택 기억 |
| 문서 프레임워크 | [VitePress](https://vitepress.dev) | 약 1.8만 스타 (2026-07) | 본문 최대 688px, ⌘K 검색 화살표 이동, 코드 그룹 탭 |
| 문서 프레임워크 | [Docusaurus](https://docusaurus.io) | 약 6.4만 스타 (2026-04) | 코드 블록 제목·복사, 이전/다음 페이저 |
| 튜토리얼 | [TutorialKit](https://tutorialkit.dev) (StackBlitz, Astro 기반) | 약 700 스타 | 지시문 40% \| 작업 공간 분할, **Solve⇄Reset 토글 하나**, '파트 / 장 / 수업' 드롭다운 내비, 모바일 하단 '설명 ⇄ 코드' 전환, 페이지 이동에도 실행 환경 유지 |
| 튜토리얼 | [svelte.dev/tutorial](https://svelte.dev/tutorial) | 프레임워크 공식 튜토리얼 | 지시문 33%(최소 360px), **정답과 같아지면 자동 완료**, 800px 미만 모바일 모드 |
| 튜토리얼 | [react.dev/learn](https://react.dev/learn) | 프레임워크 공식 문서 | 도전 문제 탭 'Challenge k of N', 힌트·해답 토글이 서로 배타, '다음 도전', 접는 'Deep Dive' |
| 설명 | [TensorFlow Playground](https://playground.tensorflow.org) | 약 1.3만 스타, 논문 발표 | 직접 조작은 '빠르고, 조금씩, 되돌릴 수 있게', 재생·한 단계·초기화, 데이터 프리셋 |
| 설명 | [CNN Explainer](https://poloclub.github.io/cnn-explainer) | 약 9천 스타, IEEE VIS 2020 | 개요 → 상세를 애니메이션으로 전환, 호버하면 수식과 그림이 함께 갱신 |
| 설명 | [Transformer Explainer](https://poloclub.github.io/transformer-explainer) | 약 8천 스타, 사용자 12.5만 | 통찰 하나에 결정적인 컨트롤 하나(temperature 슬라이더) |
| 설명 | [3Blue1Brown / manim](https://github.com/3b1b/manim) | 약 8.7만 스타 | 기저 벡터마다 고정 색, 항등에서 보간, 원래 격자를 흐리게 남김 |
| 설명 | [Distill](https://distill.pub) | ML 설명 저널 | 그림 폭 몇 가지로 제한, 수식 항과 그림의 색 연결 |
| 설명 | [Nicky Case](https://ncase.me) · [Seeing Theory](https://seeing-theory.brown.edu) | 다수 수상·보도 | 슬라이드처럼 '다음'으로 넘기는 설명, 짧은 절 + 그림 |
| 설명 | [R2D3](http://www.r2d3.us) | Information is Beautiful 2015 수상 | 스크롤텔링 — 우리 철학과 충돌하므로 **단계형으로 바꿔** 쓴다 |

## 3. 채택할 패턴

화면별로 정리한다. 괄호는 근거 레퍼런스다.

### 3.1 장 읽기 화면

1. **한 단계 = 한 개념.** 단계 막대는 숫자 없이 굵게, 지금 단계는 엣지 색 (Brilliant, Duolingo). *현재 구현 유지, 막대만 굵게.*
2. **상단 내비를 '트랙 / 장 / 단계' 드롭다운 하나로.** 누르면 그 장의 단계 목록과 같은 부의 장 목록이 애니메이션으로 펼쳐진다. 오른쪽 목차 열은 없앤다 (TutorialKit). → 본문 폭을 넓히고 화면 요소를 줄인다.
3. **본문 폭과 글자 크기 토큰:** 본문 45rem(720px), 줄간격 1.75, 13·14·16·18·20·24px 6단계 (Starlight, VitePress).
4. **하단 고정 진행 바:** 이전 · 단계 표시 · 다음을 화면 아래 한 줄에 고정하고, 지금의 floating dock과 합친다 (Duolingo 하단 버튼, TutorialKit 모바일 바).
5. **콜아웃은 4종까지**(참고·팁·주의·과학자/AI 연구자용 다리), 깊은 내용은 접는 상자 (Starlight, react.dev Deep Dive).
6. **선수 지식 칩에 호버하면 미리보기** (MyST).

### 3.2 실습(코드) 화면

7. **연습문제는 분할 화면:** 왼쪽 문제 33–40%(최소 360px) | 오른쪽 위 편집기 · 아래 테스트 결과. 800px 미만에서는 하단 '문제 ⇄ 코드' 전환 (svelte, TutorialKit, Exercism).
8. **편집기 머리에 버튼 하나: '해답 보기 ⇄ 처음으로'** (TutorialKit, svelte). 해답은 먼저 풀어 보라는 문구와 함께.
9. **테스트를 모두 통과하면 자동 완료**·진도 저장, '완료' 버튼 없음 (svelte).
10. **여러 연습문제는 'k / N: 제목' 탭 + 다음 문제 버튼**, 힌트·해답은 서로 배타 토글 (react.dev).
11. **브라우저 Python을 페이지 이동에도 유지**(View Transitions의 persist) (TutorialKit).

### 3.3 퀴즈

12. **하단 고정 주 버튼 하나:** 답을 고르기 전 비활성 → '확인' → 피드백 → '계속' (Duolingo, Brilliant). *현재는 확인·다음이 따로 있다.*
13. **오답 피드백에서 그 내용을 다룬 단계로 바로 가기** 링크 (The Odin Project).

### 3.4 위젯

14. **그림 자체를 끈다.** 슬라이더는 잡을 수 없는 값에만 (Ciechanowski, setosa).
15. **수학 객체마다 고정 색, 수식·그림·코드에서 같은 색:** 사용자가 바꾸는 것 = 엣지 색, 기저·참조 = Glacier 명도 차 (3Blue1Brown, Distill).
16. **모든 위젯에 초기화, 변화는 한 프레임 안에** (Playground).
17. **통찰 하나에 컨트롤 하나.** 예: pinball 탭에서만 τ 슬라이더를 보인다 (Transformer Explainer).
18. **단계형 그림 카드:** 스크롤텔링 대신 그림을 고정하고 1/4…4/4 단계 버튼과 두 줄 설명 (Nicky Case, R2D3를 단계형으로).
19. **상태를 주소에 담아 공유** (Playground).

### 3.5 로드맵·진도

20. **지도가 곧 진도 화면:** 노드 색이 상태(대기·학습 중·완료)를 보이고, 노드를 누르면 페이지 이동 대신 옆 패널 (roadmap.sh).
21. **장 상태 3단계**(시작 전 · 학습 중 · 완료)를 트랙까지 집계. 연속 학습일·하트·효과음은 쓰지 않는다 (Khan Academy 단순화, Duolingo의 게임 요소는 제외).

## 4. 채택하지 않는 것

| 패턴 | 출처 | 이유 |
|---|---|---|
| 스크롤텔링(스크롤에 따라 그림이 바뀜) | R2D3, MLU-Explain | 긴 스크롤 금지 원칙. 단계형 그림 카드(§3.4-18)로 바꾼다 |
| 연속 학습일·하트·효과음 | Duolingo | 기술서의 어조와 맞지 않는다 |
| IDE식 끌어서 나누는 패널 | LeetCode | 복잡하고 모바일에서 쌓이지 않는다 |
| 215개 항목 고정 사이드바 + 오른쪽 목차의 문서형 배치 | Starlight, Docusaurus | 긴 스크롤 문서 전제. 드롭다운 내비(§3.1-2)로 대신한다 |
| 프레임워크 이전(Starlight·TutorialKit·MyST) | — | 원고 로더·퀴즈·진도 키를 다시 짜야 하고, TutorialKit은 Node(WebContainer)+React 전제라 Pyodide와 맞지 않는다. **패턴만 가져온다.** 코드 블록용 Expressive Code와 검색용 Pagefind는 필요해지면 따로 도입을 검토한다 |

## 5. 적용 순서 (제안)

1. 장 화면: 상단 드롭다운 내비 + 오른쪽 목차 제거 + 본문 폭·글자 토큰 + 하단 고정 진행 바 (§3.1)
2. 퀴즈: 하단 고정 주 버튼 하나 (§3.3)
3. 연습문제: 분할 화면 + 해답⇄처음으로 토글 + 자동 완료 (§3.2)
4. 위젯 3종: 색 규칙·초기화·단계형 카드 (§3.4)
5. 지도: 상태 색 + 옆 패널 (§3.5)
6. 목차만 있는 장: 빈 화면 대신 '다룰 내용' 카드
