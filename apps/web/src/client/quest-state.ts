// 퀘스트 노드의 완료 판정(홈 진행 점과 퀘스트 화면이 함께 쓴다).
// 읽기 노드: 마지막 쪽까지 넘겨 '완료'를 누름(steps). 퀴즈: 정답. 실습실: 연습문제 모두 통과. 통과 기준: '장 완료'를 누름.
import { progress } from "./store";

export interface NodeRef { id: string; kind: string; quiz?: string }

export function nodeDone(chapterId: string, n: NodeRef, exIds: string[]): boolean {
  const p = progress.get(chapterId);
  if (n.kind === "quiz") return !!(n.quiz && p.quiz?.[n.quiz]?.correct);
  if (n.kind === "lab") return exIds.length > 0 && exIds.every((id) => p.exercises?.[id]?.passed);
  return !!p.steps?.[n.id];
}

/** 노드별 상태와 지금 할 노드(처음으로 끝나지 않은 노드) */
export function questStates(chapterId: string, nodes: NodeRef[], exIds: string[]) {
  const done = nodes.map((n) => nodeDone(chapterId, n, exIds));
  const cur = done.findIndex((d) => !d);
  return { done, cur, count: done.filter(Boolean).length };
}

export function burst(x: number, y: number, n = 32) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const C = ["#0a84ff", "#34c759", "#ff9f0a", "#8e7bff", "#ff3b30"];
  for (let i = 0; i < n; i++) {
    const p = document.createElement("i");
    p.className = "burst";
    const a = Math.random() * Math.PI * 2, d = 60 + Math.random() * 140;
    p.style.cssText = `left:${x}px;top:${y}px;background:${C[i % 5]};--dx:${(Math.cos(a) * d).toFixed(0)}px;--dy:${(Math.sin(a) * d - 40).toFixed(0)}px;--r:${(Math.random() * 540).toFixed(0)}deg`;
    document.body.append(p);
    setTimeout(() => p.remove(), 950);
  }
}
