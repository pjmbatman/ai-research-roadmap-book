# 0010. 시각·UX 설계를 처음부터 다시 하고, 레퍼런스를 그대로 따른다
- 상태: 채택
- 날짜: 2026-11-06
- 대체: [0007](0007-visual-design.md)

## 맥락
v0.1·v0.2는 처음 정한 방향을 유지했다. 그 방향은 Midnight·Glacier·Coral 테마, floating dock, 효과 도구 상자, 위젯 종류(`kind`), 캔버스 지도였다. v0.2에서 벤치마킹 패턴을 일부 덧붙였지만, 사용자 평가는 "여전히 그대로"였다. 사용자의 요구는 처음의 UI/UX 방향과 위젯·dock 명세를 모두 버리고, 인기 있는 레퍼런스를 근거로 처음부터 다시 만드는 것이다.

## 결정
- 이전 디자인 시스템(ADR 0007, `design-system.md`의 이전 판, `ux-design.md`의 화면 명세)을 폐기한다.
- 위젯 레지스트리에서 디자인 명세 필드(`kind`·`interaction`·`binds`·`motion`)를 지운다. 진도 키인 `name`과 삽입에 필요한 `title`·`chapters`·`status`만 남긴다.
- 시각 기반은 shadcn/ui v4의 neutral 토큰을 그대로 쓴다. 우리 브랜드 색은 두지 않는다.
- 화면 구조는 화면 유형마다 레퍼런스를 하나씩 정해 따른다. 공통 머리는 shadcn/ui 문서 사이트, 로드맵과 지도는 roadmap.sh, 장 읽기와 퀴즈는 Brilliant·Duolingo, 연습문제는 TutorialKit·svelte, 위젯은 TensorFlow Playground를 따른다. 자세한 대응은 [design-system.md](../design-system.md) §3에 있다.
- 사용자 원칙(심플함, 작은 글씨 없음, 스크롤 대신 넘기기 — [ADR 0009](0009-paged-reading.md))은 유지한다.

## 결과
- 사이트의 CSS·셸·홈·트랙·지도·장·위젯 화면을 새로 만든다. 원고 형식(Markdown·TOML), 진도 저장 키, 런타임(Pyodide)은 바뀌지 않는다.
- 새 화면이나 컴포넌트는 레퍼런스를 먼저 찾고, 없으면 결정을 기록한 뒤 만든다.
