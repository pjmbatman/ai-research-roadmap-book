// 원고 Markdown → HTML. 지시문(::widget, ::quiz, ::exercise, :::note), 실행 코드 블록, 기록된 output 블록,
// 고정 절(학습 목표·실습·실패 사례·통과 기준)을 사이트 컴포넌트용 HTML로 바꾼다. (editorial/conventions.md §10)
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkDirective from "remark-directive";
import remarkRehype from "remark-rehype";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import { visit } from "unist-util-visit";
import { FIXED } from "./book";

export interface RenderContext {
  assetBase?: string;   // 상대 경로 그림의 URL 접두사
  quizSection?: boolean; // 통과 기준 앞에 전체 퀴즈 자리 삽입
}

export const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const CALLOUTS = new Set(["note", "tip", "caution"]);
const FIXED_CLASS: Record<string, string> = { "학습 목표": "goals", "실습": "lab", "실패 사례": "pitfalls", "통과 기준": "pass" };

function parseMeta(meta: string | null | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  if (!meta) return out;
  for (const m of meta.matchAll(/(\w+)(?:="([^"]*)")?/g)) out[m[1]] = m[2] ?? "true";
  return out;
}

function textOf(node: any): string {
  if (!node) return "";
  if (node.type === "text" || node.type === "inlineCode") return node.value;
  return (node.children || []).map(textOf).join("");
}

function remarkBook(ctx: RenderContext) {
  return (tree: any) => {
    // 지시문
    visit(tree, (node: any, index: number | undefined, parent: any) => {
      if (node.type === "textDirective") {
        // 의도하지 않은 ":단어"는 글자로 되돌린다.
        parent.children[index!] = { type: "text", value: ":" + node.name + textOf(node) };
        return;
      }
      if (node.type === "leafDirective") {
        const a = node.attributes || {};
        let html = "";
        if (node.name === "widget") html = `<div class="widget" data-widget="${esc(a.name || "")}" data-attrs="${esc(JSON.stringify(a))}"></div>`;
        else if (node.name === "quiz") html = `<div class="quiz-slot" data-quiz="${esc(a.id || "")}"></div>`;
        else if (node.name === "exercise") html = `<div class="exercise" data-exercise="${esc(a.id || "")}"></div>`;
        else html = `<p>::${esc(node.name)}</p>`;
        parent.children[index!] = { type: "html", value: html };
        return;
      }
      if (node.type === "containerDirective" && CALLOUTS.has(node.name)) {
        const first = node.children?.[0];
        let title = "";
        if (first?.data?.directiveLabel) {
          title = textOf(first);
          node.children.shift();
        }
        node.data = { hName: "aside", hProperties: { className: ["callout", `callout-${node.name}`], "data-title": title } };
        if (title) node.children.unshift({ type: "paragraph", data: { hName: "p", hProperties: { className: ["callout-title"] } }, children: [{ type: "text", value: title }] });
      }
    });

    // 실행 코드 블록과 기록된 출력
    visit(tree, (node: any) => {
      if (!Array.isArray(node.children)) return;
      const kids = node.children;
      for (let i = 0; i < kids.length; i++) {
        const k = kids[i];
        if (k.type !== "code") continue;
        const meta = parseMeta(k.meta);
        if (meta.exec) {
          let out = "";
          const nxt = kids[i + 1];
          if (nxt && nxt.type === "code" && nxt.lang === "output") {
            out = nxt.value;
            kids.splice(i + 1, 1);
          }
          kids[i] = {
            type: "html",
            value:
              `<div class="cell" data-exec="${esc(meta.exec)}" data-session="${esc(meta.session || "default")}" data-lang="${esc(k.lang || "python")}">` +
              `<div class="cell-bar"><span class="badge badge-exec">${esc(meta.exec)}</span><span class="cell-lang">${esc(k.lang || "")}</span></div>` +
              `<pre class="cell-src"><code>${esc(k.value)}</code></pre>` +
              (out ? `<pre class="cell-out" data-recorded="true">${esc(out)}</pre>` : `<pre class="cell-out" hidden></pre>`) +
              `</div>`,
          };
        } else if (k.lang === "output") {
          kids[i] = { type: "html", value: `<pre class="output">${esc(k.value)}</pre>` };
        }
      }
    });

    // 상대 경로 그림
    visit(tree, "image", (node: any) => {
      if (ctx.assetBase && !/^([a-z]+:|\/)/.test(node.url)) node.url = ctx.assetBase + node.url;
    });
  };
}

function rehypeSections(ctx: RenderContext) {
  return (tree: any) => {
    const out: any[] = [];
    let current: any = null;
    const flush = () => { if (current) out.push(current); current = null; };
    for (const node of tree.children) {
      if (node.type === "element" && node.tagName === "h2") {
        flush();
        const title = textOf(node);
        if (title === "통과 기준" && ctx.quizSection) {
          out.push({
            type: "element", tagName: "section", properties: { className: ["box", "box-quiz"], id: "quiz-all-section" },
            children: [
              { type: "element", tagName: "h2", properties: { id: "개념-퀴즈" }, children: [{ type: "text", value: "개념 퀴즈" }] },
              { type: "element", tagName: "div", properties: { className: ["quiz-all"] }, children: [] },
            ],
          });
        }
        const cls = FIXED.includes(title) ? ["box", `box-${FIXED_CLASS[title]}`] : ["concept"];
        current = { type: "element", tagName: "section", properties: { className: cls }, children: [node] };
      } else if (current) {
        current.children.push(node);
      } else {
        out.push(node);
      }
    }
    flush();
    tree.children = out;
  };
}

function processor(ctx: RenderContext, sections: boolean) {
  const p = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkDirective)
    .use(remarkBook, ctx)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeKatex, { output: "htmlAndMathml", strict: "ignore" })
    .use(rehypeSlug);
  if (sections) p.use(rehypeSections, ctx);
  return p.use(rehypeStringify, { allowDangerousHtml: true });
}

/** 장 본문 전체. 첫 h1은 페이지 머리에서 따로 그리므로 뺀다. */
export function renderChapter(md: string, ctx: RenderContext = {}): string {
  const body = md.replace(/^# .+\n/m, "");
  return String(processor(ctx, true).processSync(body));
}

export function renderMarkdown(md: string, ctx: RenderContext = {}): string {
  return String(processor(ctx, false).processSync(md));
}

/** 한 줄짜리 Markdown(퀴즈 문항·선택지). 바깥 <p>를 벗긴다. */
export function renderInline(md: string): string {
  const html = String(processor({}, false).processSync(md)).trim();
  const m = /^<p>([\s\S]*)<\/p>$/.exec(html);
  return m && !m[1].includes("<p>") ? m[1] : html;
}
