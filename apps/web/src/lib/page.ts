// 장 페이지에 심을 데이터(JSON): 퀴즈·연습문제·위젯 정보. 클라이언트 컴포넌트가 이 데이터로 동작한다.
import type { Chapter } from "./book";
import { widgetsFor } from "./book";
import { renderInline, renderMarkdown } from "./markdown";

export function chapterData(ch: Chapter) {
  const objectives = (/^## 학습 목표\s*\n([\s\S]*?)(?=^## |$(?![\s\S]))/m.exec(ch.body)?.[1] ?? "")
    .split("\n").filter((l) => l.startsWith("- ")).map((l) => l.slice(2).trim());
  const quiz = ch.quiz.map((q) => ({
    id: q.id, type: q.type, objective: q.objective, objectiveText: objectives[q.objective - 1] ?? "",
    prompt: renderInline(q.prompt), explanation: renderInline(q.explanation),
    choices: (q.choices ?? []).map((c) => ({ html: renderInline(c.text), correct: !!c.correct, feedback: c.feedback ? renderInline(c.feedback) : "" })),
    answer: q.answer, tolerance: q.tolerance ?? 0, slider: q.slider ?? null,
  }));
  const exercises = Object.fromEntries(
    Object.values(ch.exercises).map((e) => [e.id, { ...e, problem: renderMarkdown(e.problem.replace(/^# .+\n/m, ""), { fromDir: ch.dir }), titleText: e.title }]),
  );
  const widgets = Object.fromEntries(widgetsFor(ch).map((w) => [w.name, { title: w.title, kind: w.kind, status: w.status }]));
  return { chapterId: ch.id, title: String(ch.meta.title ?? ""), objectives, quiz, exercises, widgets };
}
