// 로그인 없는 진도 저장소(브라우저 localStorage)와 페이지 상태 버스.
// 키는 장 id·퀴즈 문항 id·연습문제 id이므로 원고의 영구 id가 바뀌지 않는 한 유지된다.

export interface ChapterProgress {
  visited?: number;
  quiz?: Record<string, { correct: boolean; tries: number }>;
  exercises?: Record<string, { passed: boolean; code?: string }>;
  checks?: Record<string, boolean>;
}
interface Saved { version: 1; chapters: Record<string, ChapterProgress> }

const KEY = "roadmap:progress:v1";
let memory: Saved = { version: 1, chapters: {} };

function read(): Saved {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) memory = JSON.parse(raw);
  } catch {}
  if (!memory.chapters) memory = { version: 1, chapters: {} };
  return memory;
}
function write() {
  try { localStorage.setItem(KEY, JSON.stringify(memory)); } catch {}
  window.dispatchEvent(new CustomEvent("progress-change"));
}

export const progress = {
  all(): Record<string, ChapterProgress> { return read().chapters; },
  get(id: string): ChapterProgress { return read().chapters[id] ?? {}; },
  update(id: string, fn: (p: ChapterProgress) => void) {
    const s = read();
    const p = (s.chapters[id] ??= {});
    fn(p);
    write();
  },
  export(): string { return JSON.stringify({ ...read(), exportedAt: new Date().toISOString() }, null, 2); },
  import(text: string) {
    const data = JSON.parse(text);
    if (!data || typeof data.chapters !== "object") throw new Error("진도 파일 형식이 아닙니다");
    memory = { version: 1, chapters: data.chapters };
    write();
  },
  reset() { memory = { version: 1, chapters: {} }; write(); },
};

/** 장의 진도율: 퀴즈 정답 + 연습문제 통과 + 통과 기준 체크 / 전체 */
export function chapterScore(id: string, totals: { quiz: number; exercises: number; checks: number }) {
  const p = progress.get(id);
  const q = Object.values(p.quiz ?? {}).filter((x) => x.correct).length;
  const e = Object.values(p.exercises ?? {}).filter((x) => x.passed).length;
  const c = Object.values(p.checks ?? {}).filter(Boolean).length;
  const total = totals.quiz + totals.exercises + totals.checks;
  return { done: q + e + c, total, ratio: total ? (q + e + c) / total : 0 };
}

/** 같은 페이지의 위젯·수식·코드 셀이 공유하는 상태 */
class StateBus extends EventTarget {
  private values = new Map<string, unknown>();
  get<T = unknown>(k: string): T | undefined { return this.values.get(k) as T | undefined; }
  set(k: string, v: unknown, source?: unknown) {
    this.values.set(k, v);
    this.dispatchEvent(new CustomEvent("change", { detail: { key: k, value: v, source } }));
  }
  snapshot(): Record<string, unknown> { return Object.fromEntries(this.values); }
}
export const pageState = new StateBus();
