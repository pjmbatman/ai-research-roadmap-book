// 장 본문 → 퀘스트(ADR 0012). 절(##) 하나가 노드 하나이고, 본문 속 ::quiz는 그 절 바로 뒤의 퀴즈 노드가 된다.
// 본문에 걸리지 않은 퀴즈는 '통과 기준' 앞에 모은다. '실습'은 연습문제가 있으면 실습실 노드, '통과 기준'은 마지막 관문.
// 노드가 많으면 다섯 개 이하씩 구간(탭)으로 나눈다. 노드 id는 진도 기록의 키이므로 절 제목과 퀴즈 id에서 만든다.
import GithubSlugger from "github-slugger";
import type { Chapter } from "./book";

export type NodeKind = "goal" | "read" | "widget" | "code" | "quiz" | "lab" | "pitfall" | "ref" | "boss";
export interface QuestNode { id: string; kind: NodeKind; name: string; section?: number; quiz?: string; objective?: number }
export interface QuestGroup { name: string; nodes: string[] }
export interface Quest { nodes: QuestNode[]; groups: QuestGroup[] }

const GROUP_MAX = 5;

/** 코드 펜스 밖의 ## 절을 순서대로 나눈다 */
export function h2Sections(body: string): { title: string; text: string }[] {
  const out: { title: string; text: string }[] = [];
  let fence = "";
  let cur: { title: string; lines: string[] } | null = null;
  for (const line of body.split("\n")) {
    const f = /^\s*(```+|~~~+)/.exec(line);
    if (f) fence = fence ? (line.trim().startsWith(fence) ? "" : fence) : f[1];
    if (!fence && /^## /.test(line)) {
      if (cur) out.push({ title: cur.title, text: cur.lines.join("\n") });
      cur = { title: line.slice(3).trim(), lines: [] };
    } else if (cur) cur.lines.push(line);
  }
  if (cur) out.push({ title: cur.title, text: cur.lines.join("\n") });
  return out;
}

export function questOf(ch: Chapter): Quest {
  const slug = new GithubSlugger();
  const quizById = new Map(ch.quiz.map((q) => [q.id, q]));
  const placed = new Set<string>();
  const hasLab = Object.keys(ch.exercises).length > 0;
  const nodes: QuestNode[] = [];
  let boss: QuestNode | null = null;
  const refs: QuestNode[] = [];
  let qn = 0;
  const quizNode = (id: string): QuestNode => {
    placed.add(id);
    return { id: `q:${id}`, kind: "quiz", name: `퀴즈 ${++qn}`, quiz: id, objective: quizById.get(id)?.objective };
  };
  const leftovers = () => ch.quiz.filter((q) => !placed.has(q.id)).map((q) => quizNode(q.id));

  h2Sections(ch.body).forEach((s, i) => {
    const id = `s:${slug.slug(s.title)}`;
    if (s.title === "통과 기준") { boss = { id, kind: "boss", name: "통과 기준", section: i }; return; }
    if (s.title === "참고 문헌") { refs.push({ id, kind: "ref", name: "참고 문헌", section: i }); return; }
    const kind: NodeKind =
      s.title === "학습 목표" ? "goal"
      : s.title === "실습" ? (hasLab ? "lab" : "code")
      : s.title === "실패 사례" ? "pitfall"
      : /::widget\{/.test(s.text) ? "widget"
      : /exec="/.test(s.text) ? "code"
      : "read";
    nodes.push({ id, kind, name: s.title, section: i });
    for (const m of s.text.matchAll(/::quiz\{id="([^"]+)"\}/g)) if (quizById.has(m[1]) && !placed.has(m[1])) nodes.push(quizNode(m[1]));
  });
  nodes.push(...leftovers(), ...refs);
  if (boss) nodes.push(boss);

  // 구간: 절과 바로 뒤의 퀴즈를 한 덩어리로 묶고, 덩어리째 다섯 개 이하로 고르게 나눈다
  const units: QuestNode[][] = [];
  for (const n of nodes) {
    const last = units[units.length - 1];
    if (n.kind === "quiz" && last && (last[0].kind !== "quiz" || last.every((x) => x.kind === "quiz"))) last.push(n);
    else units.push([n]);
  }
  const target = Math.ceil(nodes.length / Math.max(1, Math.ceil(nodes.length / GROUP_MAX)));
  const packs: QuestNode[][] = [];
  for (const u of units) {
    const cur = packs[packs.length - 1];
    if (cur && cur.length + u.length <= target) cur.push(...u);
    else packs.push([...u]);
  }
  // 끝에 한두 개만 남은 구간은 앞 구간에 붙인다(다섯 개를 넘지 않을 때)
  while (packs.length > 1 && packs[packs.length - 1].length <= 2 && packs[packs.length - 2].length + packs[packs.length - 1].length <= GROUP_MAX) {
    const tail = packs.pop()!;
    packs[packs.length - 1].push(...tail);
  }
  const groups: QuestGroup[] = packs.map((part, i) => {
    const named = part.find((n) => n.kind !== "quiz") ?? part[0];
    const name = i === 0 ? "시작" : part.some((n) => n.kind === "boss") ? "마무리" : named.name;
    return { name, nodes: part.map((n) => n.id) };
  });
  return { nodes, groups };
}
