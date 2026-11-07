// 여러 페이지가 함께 쓰는 원고 요약: 장의 학습 목표·항목 수, 트랙 사이의 의존 관계.
import { loadBook, allChapters, str, partOf, trackOfChapter, type Chapter } from "./book";
import { questOf } from "./quest";
import { renderInline } from "./markdown";

const section = (body: string, title: string) =>
  (new RegExp(`^## ${title}\\s*\\n([\\s\\S]*?)(?=^## |$(?![\\s\\S]))`, "m").exec(body)?.[1] ?? "")
    .split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2).trim());

export const goalsOf = (c: Chapter) => section(c.body, "학습 목표");
export const passCount = (c: Chapter) => section(c.body, "통과 기준").length;
export const isReady = (c: Chapter) => ["draft", "review", "done"].includes(str(c.meta.status));
export const trackKeyOf = (c: Chapter) => str(loadBook().parts.find((p) => p.slug === c.partSlug)?.meta.track);
export const trackCodeOf = (key: string) => ((loadBook().config.tracks as any[]).find((t) => t.key === key)?.code as string) ?? "REF";

/** 클라이언트가 진도 상태를 계산하는 데 필요한 장 요약 */
export function chapterSummary(c: Chapter) {
  return {
    id: c.id, label: c.label, title: str(c.meta.title), url: c.url, ready: isReady(c), tier: str(c.meta.tier),
    track: trackKeyOf(c), code: trackCodeOf(trackKeyOf(c)), goals: goalsOf(c), pre: (c.meta.prerequisites as string[]) || [],
    quiz: c.quiz.length, ex: Object.keys(c.exercises).length, checks: passCount(c),
    exIds: Object.keys(c.exercises),
    /** 본문이 있는 장의 퀘스트 노드(홈의 진행 점) */
    nodes: isReady(c) ? questOf(c).nodes.map((n) => ({ id: n.id, kind: n.kind, quiz: n.quiz, name: n.name })) : [],
    /** 본문이 있는 장의 도입 문단(홈 히어로) */
    lead: isReady(c) ? leadOf(c) : "",
    crumb: `${trackOfChapter(c)?.name ?? "부록"} › ${str(partOf(c).meta.title)}`,
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

/** 첫 절 앞의 도입 문단 첫 단락(HTML) */
export function leadOf(c: Chapter): string {
  const body = c.body.replace(/^# .+\n/m, "");
  const lead = body.split(/^## /m)[0].trim().split(/\n\s*\n/)[0] ?? "";
  return lead ? renderInline(lead) : "";
}
