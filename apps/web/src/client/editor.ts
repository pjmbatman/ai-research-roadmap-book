// CodeMirror 6 편집기 공통 설정. 색은 디자인 토큰(--code, --foreground)과 GitHub 기본 테마 문법 색을 따른다.
import { EditorView, keymap, lineNumbers, highlightActiveLine, drawSelection } from "@codemirror/view";
import { EditorState, type Extension } from "@codemirror/state";
import { history, defaultKeymap, historyKeymap, indentWithTab } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { HighlightStyle, syntaxHighlighting, indentUnit, bracketMatching } from "@codemirror/language";
import { tags as t } from "@lezer/highlight";

// 문법 색은 CSS 변수(--syn-*)로 두어 라이트·다크 테마를 따른다(GitHub 기본 테마 색, global.css).
const style = HighlightStyle.define([
  { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword, t.operatorKeyword], color: "var(--syn-keyword)" },
  { tag: [t.string, t.special(t.string)], color: "var(--syn-string)" },
  { tag: [t.number, t.bool, t.null], color: "var(--syn-number)" },
  { tag: [t.comment, t.lineComment], color: "var(--syn-comment)", fontStyle: "italic" },
  { tag: [t.function(t.variableName), t.function(t.propertyName)], color: "var(--syn-function)" },
  { tag: [t.definition(t.variableName), t.definition(t.function(t.variableName))], color: "var(--syn-function)" },
  { tag: t.className, color: "var(--syn-class)" },
]);

const theme = EditorView.theme({
  "&": { backgroundColor: "var(--code)", color: "var(--foreground)" },
  ".cm-content": { caretColor: "var(--foreground)", padding: "12px 0", fontFamily: "var(--mono)" },
  ".cm-cursor": { borderLeftColor: "var(--foreground)" },
  ".cm-gutters": { backgroundColor: "var(--code)", color: "var(--muted-foreground)", border: "none" },
  ".cm-activeLine": { backgroundColor: "color-mix(in srgb, var(--foreground) 4%, transparent)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--foreground)" },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": { backgroundColor: "color-mix(in srgb, var(--chart-1) 45%, transparent) !important" },
  ".cm-matchingBracket": { backgroundColor: "var(--muted)", outline: "none" },
});

export function createEditor(parent: HTMLElement, doc: string, opts: { onRun?: () => void; onChange?: (s: string) => void; label?: string; readOnly?: boolean } = {}) {
  const ext: Extension[] = [
    lineNumbers(), history(), drawSelection(), highlightActiveLine(), bracketMatching(), indentUnit.of("    "),
    python(), syntaxHighlighting(style), theme,
    keymap.of([
      { key: "Shift-Enter", run: () => { opts.onRun?.(); return true; } },
      { key: "Mod-Enter", run: () => { opts.onRun?.(); return true; } },
      ...defaultKeymap, ...historyKeymap, indentWithTab,
    ]),
    EditorView.contentAttributes.of({ "aria-label": opts.label ?? "Python 코드 편집기" }),
  ];
  if (opts.readOnly) ext.push(EditorState.readOnly.of(true), EditorView.editable.of(false));
  if (opts.onChange) ext.push(EditorView.updateListener.of((u) => { if (u.docChanged) opts.onChange!(u.state.doc.toString()); }));
  const view = new EditorView({ state: EditorState.create({ doc, extensions: ext }), parent });
  return {
    view,
    get: () => view.state.doc.toString(),
    set: (s: string) => view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: s } }),
  };
}
