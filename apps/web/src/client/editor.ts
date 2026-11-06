// CodeMirror 6 편집기 공통 설정. 코드 배경은 테마와 상관없이 Midnight을 쓴다(design-system.md).
import { EditorView, keymap, lineNumbers, highlightActiveLine, drawSelection } from "@codemirror/view";
import { EditorState, type Extension } from "@codemirror/state";
import { history, defaultKeymap, historyKeymap, indentWithTab } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { HighlightStyle, syntaxHighlighting, indentUnit, bracketMatching } from "@codemirror/language";
import { tags as t } from "@lezer/highlight";

const style = HighlightStyle.define([
  { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword, t.operatorKeyword], color: "#ff8f80" },
  { tag: [t.string, t.special(t.string)], color: "#a8d8b9" },
  { tag: [t.number, t.bool, t.null], color: "#f5c27a" },
  { tag: [t.comment, t.lineComment], color: "#7f8aa3", fontStyle: "italic" },
  { tag: [t.function(t.variableName), t.function(t.propertyName)], color: "#9fc6ff" },
  { tag: [t.definition(t.variableName), t.definition(t.function(t.variableName))], color: "#dce6f2", fontWeight: "600" },
  { tag: t.className, color: "#c9b6ff" },
  { tag: t.operator, color: "#b9c6d8" },
]);

const theme = EditorView.theme(
  {
    "&": { backgroundColor: "#0b1020", color: "#dce6f2" },
    ".cm-content": { caretColor: "#ff6b5a", padding: "12px 0" },
    ".cm-cursor": { borderLeftColor: "#ff6b5a" },
    ".cm-gutters": { backgroundColor: "#0b1020", color: "#4d5874", border: "none" },
    ".cm-activeLine": { backgroundColor: "rgba(220,230,242,0.04)" },
    ".cm-activeLineGutter": { backgroundColor: "transparent", color: "#8d9ab0" },
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": { backgroundColor: "rgba(255,107,90,0.25) !important" },
    ".cm-matchingBracket": { backgroundColor: "rgba(220,230,242,0.12)", outline: "none" },
  },
  { dark: true },
);

export function createEditor(parent: HTMLElement, doc: string, opts: { onRun?: () => void; onChange?: (s: string) => void; label?: string } = {}) {
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
  if (opts.onChange) ext.push(EditorView.updateListener.of((u) => { if (u.docChanged) opts.onChange!(u.state.doc.toString()); }));
  const view = new EditorView({ state: EditorState.create({ doc, extensions: ext }), parent });
  return {
    view,
    get: () => view.state.doc.toString(),
    set: (s: string) => view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: s } }),
  };
}
