// 여러 페이지가 함께 쓰는 원고 요약: 장의 학습 목표·항목 수, 트랙 사이의 의존 관계.
import { loadBook, allChapters, str, type Chapter } from "./book";

const section = (body: string, title: string) =>
  (new RegExp(`^## ${title}\\s*\\n([\\s\\S]*?)(?=^## |$(?![\\s\\S]))`, "m").exec(body)?.[1] ?? "")
    .split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2).trim());

export const goalsOf = (c: Chapter) => section(c.body, "학습 목표");
export const passCount = (c: Chapter) => section(c.body, "통과 기준").length;
export const isReady = (c: Chapter) => ["draft", "review", "done"].includes(str(c.meta.status));
export const trackKeyOf = (c: Chapter) => str(loadBook().parts.find((p) => p.slug === c.partSlug)?.meta.track);

/** 클라이언트가 진도 상태를 계산하는 데 필요한 장 요약 */
export function chapterSummary(c: Chapter) {
  return {
    id: c.id, label: c.label, title: str(c.meta.title), url: c.url, ready: isReady(c), tier: str(c.meta.tier),
    track: trackKeyOf(c), goals: goalsOf(c), pre: (c.meta.prerequisites as string[]) || [],
    quiz: c.quiz.length, ex: Object.keys(c.exercises).length, checks: passCount(c),
  };
}
export type ChapterSummary = ReturnType<typeof chapterSummary>;

/** 트랙 A → B: B의 장이 A의 장을 선수 지식으로 둔다 (같은 트랙 안은 제외) */
export function trackEdges(): [string, string][] {
  const byId = new Map(allChapters().map((c) => [c.id, c]));
  const edges = new Set<string>();
  for (const c of allChapters()) {
    const to = trackKeyOf(c);
    for (const p of (c.meta.prerequisites as string[]) || []) {
      const from = byId.get(p) && trackKeyOf(byId.get(p)!);
      if (from && to && from !== to) edges.add(`${from}>${to}`);
    }
  }
  return [...edges].map((e) => e.split(">") as [string, string]);
}
