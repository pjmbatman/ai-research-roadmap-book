// CLI 인자에서 설정으로 (data/widgets.toml: argparse-sandbox, 6.4장 "argparse"·"설정과 CLI의 병합").
// 실습 soh_cli.py의 train 명령을 단순화해 TypeScript로 옮겼다. 명령행 프리셋 하나를 고르면
//   1) argparse 해석(옵션 이름·접두사, 값 개수, type 변환, choices) → Namespace 또는 오류
//   2) 병합(기본값 < --config 파일 < 개별 옵션·--set)과 검증(모르는 설정 키, 타입) → 최종 설정과 출처 또는 오류
// 를 터미널처럼 보인다. 스위치 두 개는 본문이 말한 조용한 함정을 켜고 끈다.
//   'parser 기본값 None'을 끄면 옵션의 parser 기본값이 기본 설정 값이 되어, 주지 않은 옵션도 설정 파일을 덮어쓴다.
//   'allow_abbrev=False'를 끄면 옵션 이름의 고유한 접두사(--epoch → --epochs)를 받아들인다.
// 오류 문구는 Python 3.13의 argparse 문구를 따른다(집필 시점 예시, 패치 버전마다 조금 다를 수 있다).
import { frame, presets, readouts, el, C } from "../frame";

type Val = string | number | boolean;
export const DEFAULTS: Record<string, Val> = {
  "data.name": "sine", "data.n_train": 200, "data.noise": 0.1,
  "model.name": "poly", "model.degree": 5,
  "train.lr": 0.1, "train.epochs": 2000, "train.seed": 0,
};
/** configs/linear.toml (실습 파일) */
const FILES: Record<string, Record<string, Val>> = { "configs/linear.toml": { "model.name": "linear", "train.lr": 0.05, "train.seed": 3 } };
const FLOATS = new Set(["data.noise", "train.lr"]); // 기본값이 float인 키(0.1 등). JS에는 int/float 구분이 없어 따로 적는다

type Opt = { flag: string; dest: string; kind: "path" | "str" | "flag" | "int" | "float" | "append"; choices?: string[] };
const OPTS: Opt[] = [
  { flag: "--config", dest: "config", kind: "path" },
  { flag: "--out", dest: "out", kind: "path" },
  { flag: "--dry-run", dest: "dry_run", kind: "flag" },
  { flag: "--data", dest: "data.name", kind: "str", choices: ["sine", "linear"] },
  { flag: "--model", dest: "model.name", kind: "str", choices: ["poly", "linear"] },
  { flag: "--degree", dest: "model.degree", kind: "int" },
  { flag: "--lr", dest: "train.lr", kind: "float" },
  { flag: "--epochs", dest: "train.epochs", kind: "int" },
  { flag: "--seed", dest: "train.seed", kind: "int" },
  { flag: "--set", dest: "sets", kind: "append" },
];

/** Python 값 → repr 문자열(float는 소수점을 남긴다) */
export function pyRepr(v: Val | undefined, isFloat = false): string {
  if (v === undefined || v === null) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "string") return `'${v}'`;
  if (!Number.isFinite(v)) return Number.isNaN(v) ? "nan" : v > 0 ? "inf" : "-inf";
  if (isFloat || !Number.isInteger(v)) {
    if (Number.isInteger(v) && Math.abs(v) < 1e16) return `${v}.0`;
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-4 || a >= 1e16)) return v.toExponential().replace(/e([+-])(\d)$/, "e$10$2").replace("e+", "e+");
    return String(v);
  }
  return String(v);
}
const FLOAT_RE = /^[+-]?((\d(_?\d)*)?\.?\d(_?\d)*([eE][+-]?\d(_?\d)*)?|\d(_?\d)*\.|inf|infinity|nan)$/i;
const INT_RE = /^[+-]?\d(_?\d)*$/;
/** --set 값: TOML 문법으로 해석하고, 안 되면 문자열(soh_cli.parse_set) */
export function tomlValue(raw: string): { v: Val; float: boolean } {
  const s = raw.trim();
  if (/^[+-]?(0|[1-9](_?\d)*)$/.test(s)) return { v: Number(s.replace(/_/g, "")), float: false };
  if (/^[+-]?(0|[1-9](_?\d)*)(\.\d(_?\d)*)?([eE][+-]?\d(_?\d)*)?$/.test(s) && /[.eE]/.test(s)) return { v: Number(s.replace(/_/g, "")), float: true };
  if (/^[+-]?(inf|nan)$/.test(s)) return { v: s.includes("nan") ? NaN : s.startsWith("-") ? -Infinity : Infinity, float: true };
  if (s === "true" || s === "false") return { v: s === "true", float: false };
  const q = /^"([^"\\]*)"$/.exec(s) ?? /^'([^']*)'$/.exec(s);
  if (q) return { v: q[1], float: false };
  return { v: raw, float: false };
}

export interface Result {
  tokens: { text: string; role: "opt" | "val" | "bad"; dest?: string }[];
  ns: Record<string, Val | string[] | undefined> | null;
  error: string | null;     // 전체 오류 줄(“soh_cli train: error: …”)
  stage: "parse" | "merge" | null;
  cfg: Record<string, Val> | null;
  floats: Set<string>;
  sources: Record<string, string>;
  dryRun: boolean;
}

export function run(argv: string[], o: { noneDefaults: boolean; noAbbrev: boolean }): Result {
  const tokens: Result["tokens"] = [];
  const ns: Record<string, Val | string[] | undefined> = {};
  for (const op of OPTS) ns[op.dest] = op.kind === "flag" ? false : op.kind === "append" ? [] : !o.noneDefaults && op.dest in DEFAULTS ? DEFAULTS[op.dest] : undefined;
  const floats = new Set<string>();
  if (!o.noneDefaults) for (const k of FLOATS) floats.add(k);
  const extras: string[] = [];
  const fail = (msg: string, top = false): Result => ({ tokens, ns: null, error: `${top ? "soh_cli" : "soh_cli train"}: error: ${msg}`, stage: "parse", cfg: null, floats, sources: {}, dryRun: false });
  for (let i = 0; i < argv.length; i++) {
    const tok = argv[i];
    if (!tok.startsWith("-") || tok === "-") { extras.push(tok); tokens.push({ text: tok, role: "bad" }); continue; }
    const [name, inline] = tok.includes("=") ? [tok.slice(0, tok.indexOf("=")), tok.slice(tok.indexOf("=") + 1)] : [tok, undefined];
    let op = OPTS.find((p) => p.flag === name);
    if (!op && !o.noAbbrev && name.startsWith("--")) {
      const cand = OPTS.filter((p) => p.flag.startsWith(name));
      if (cand.length > 1) { tokens.push({ text: tok, role: "bad" }); return fail(`ambiguous option: ${name} could match ${cand.map((c) => c.flag).join(", ")}`); }
      if (cand.length === 1) op = cand[0];
    }
    if (!op) { extras.push(tok); tokens.push({ text: tok, role: "bad" }); continue; }
    tokens.push({ text: inline === undefined ? tok : name, role: "opt", dest: op.dest });
    if (op.kind === "flag") {
      if (inline !== undefined) { return fail(`argument ${op.flag}: ignored explicit argument '${inline}'`); }
      ns[op.dest] = true;
      continue;
    }
    let raw = inline;
    if (raw === undefined) {
      const nxt = argv[i + 1];
      if (nxt === undefined || (nxt.startsWith("-") && !/^-\d/.test(nxt))) return fail(`argument ${op.flag}: expected one argument`);
      raw = nxt; i++;
    }
    tokens.push({ text: raw, role: "val" });
    let v: Val = raw;
    if (op.kind === "int") {
      if (!INT_RE.test(raw.trim())) { tokens[tokens.length - 1].role = "bad"; return fail(`argument ${op.flag}: invalid int value: '${raw}'`); }
      v = Number(raw.trim().replace(/_/g, ""));
    } else if (op.kind === "float") {
      if (!FLOAT_RE.test(raw.trim())) { tokens[tokens.length - 1].role = "bad"; return fail(`argument ${op.flag}: invalid float value: '${raw}'`); }
      const t = raw.trim().toLowerCase().replace(/_/g, "");
      v = /nan/.test(t) ? NaN : /inf/.test(t) ? (t.startsWith("-") ? -Infinity : Infinity) : Number(t);
      floats.add(op.dest);
    }
    if (op.choices && !op.choices.includes(String(v))) { tokens[tokens.length - 1].role = "bad"; return fail(`argument ${op.flag}: invalid choice: '${raw}' (choose from ${op.choices.map((c) => `'${c}'`).join(", ")})`); }
    if (op.kind === "append") (ns[op.dest] as string[]).push(raw);
    else ns[op.dest] = v;
  }
  if (extras.length) return { ...fail(`unrecognized arguments: ${extras.join(" ")}`, true), ns };

  // 병합: soh_cli.main → resolve
  const mfail = (msg: string): Result => ({ tokens, ns, error: `soh_cli train: error: ${msg}`, stage: "merge", cfg: null, floats, sources: {}, dryRun: false });
  let file: Record<string, Val> = {};
  if (ns.config !== undefined) {
    const f = FILES[String(ns.config)];
    if (!f) return mfail(`[Errno 2] No such file or directory: '${ns.config}'`);
    file = f;
  }
  const cli: Record<string, Val> = {};
  const cliFloat = new Set<string>();
  for (const op of OPTS) if (op.dest.includes(".") && ns[op.dest] !== undefined) { cli[op.dest] = ns[op.dest] as Val; if (floats.has(op.dest)) cliFloat.add(op.dest); }
  for (const item of ns.sets as string[]) {
    const eq = item.indexOf("=");
    if (eq < 0) return mfail(`--set은 KEY=VALUE 형식이어야 합니다: '${item}'`);
    const { v, float } = tomlValue(item.slice(eq + 1));
    const key = item.slice(0, eq);
    cli[key] = v;
    if (float) cliFloat.add(key); else cliFloat.delete(key);
  }
  const cfg: Record<string, Val> = { ...DEFAULTS };
  const outFloats = new Set(FLOATS);
  const sources: Record<string, string> = Object.fromEntries(Object.keys(DEFAULTS).map((k) => [k, "default"]));
  for (const [source, flat, fl] of [["file", file, new Set(["train.lr"])], ["cli", cli, cliFloat]] as [string, Record<string, Val>, Set<string>][]) {
    for (const [k, v0] of Object.entries(flat)) {
      if (!(k in DEFAULTS)) return mfail(`알 수 없는 설정 키 '${k}' (${source})`);
      const expected = FLOATS.has(k) ? "float" : typeof DEFAULTS[k] === "string" ? "str" : "int";
      const got = typeof v0 === "string" ? "str" : typeof v0 === "boolean" ? "bool" : fl.has(k) ? "float" : "int";
      let v = v0;
      if (expected === "float" && got === "int") v = Number(v0);
      else if (expected !== got) return mfail(`${k}: ${expected}여야 합니다. 받은 값 ${pyRepr(v0, got === "float")} (${source})`);
      cfg[k] = v;
      sources[k] = source;
    }
  }
  return { tokens, ns, error: null, stage: null, cfg, floats: outFloats, sources, dryRun: ns.dry_run === true };
}
/** --dry-run 출력 줄: f"{key:14s} = {value!r:8s} ({source})" */
export function dryLines(r: Result) {
  if (!r.cfg) return [];
  return Object.keys(r.cfg).sort().map((k) => `${k.padEnd(14)} = ${pyRepr(r.cfg![k], r.floats.has(k)).padEnd(8)} (${r.sources[k]})`);
}

/** 명령행에서 실제로 준 설정 키(옵션 dest와 --set 키) */
export const given = (r: Result) => new Set([
  ...r.tokens.filter((t) => t.role === "opt" && t.dest && t.dest.includes(".")).map((t) => t.dest!),
  ...((r.ns?.sets as string[] | undefined) ?? []).map((s) => s.split("=")[0]),
]);

const PRESETS: { key: string; label: string; argv: string }[] = [
  { key: "dry", label: "--config · --seed 1", argv: "--config configs/linear.toml --seed 1 --dry-run" },
  { key: "epochs", label: "--config · --epochs 20", argv: "--config configs/linear.toml --epochs 20 --dry-run" },
  { key: "choice", label: "--model mlp", argv: "--model mlp --dry-run" },
  { key: "type", label: "--lr fast", argv: "--lr fast" },
  { key: "abbrev", label: "--epoch 5", argv: "--config configs/linear.toml --epoch 5 --dry-run" },
  { key: "typo", label: "--set 오타", argv: "--set train.learning_rate=0.3 --dry-run" },
  { key: "settype", label: "--set epochs=1e3", argv: "--set train.epochs=1e3 --dry-run" },
  { key: "set", label: "--lr · --set noise", argv: "--lr 3e-4 --set data.noise=0.2 --dry-run" },
];

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  // 터미널 줄 수(명령 1–3줄, Namespace 2–3줄, 설정 8줄)에 맞춰 캔버스 높이를 고른다
  const f = frame(root, "CLI 인자에서 설정으로", "manipulate", (root.clientWidth < 480 ? 380 : 350) / Math.max(root.clientWidth, 300),
    "명령행 토큰, argparse 해석 결과, 병합된 최종 설정과 출처 또는 오류 메시지를 터미널처럼 보인 그림");
  let cur = PRESETS.find((p) => p.key === attrs.preset) ?? PRESETS[0];
  const opts = { noneDefaults: true, noAbbrev: true };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const r = run(cur.argv.split(" "), opts);
    const fs = W < 480 ? 11 : 12.5;
    const lh = fs * 1.55;
    const x0 = 12;
    let y = 22;
    ctx.font = `${fs}px JetBrains Mono, monospace`;
    // 1) 명령행: 줄바꿈하며 토큰을 색으로 구분
    const words: { text: string; color: string; bold?: boolean; under?: boolean }[] = [
      { text: "$", color: C.mid }, { text: "soh_cli", color: C.ink }, { text: "train", color: C.ink },
      ...r.tokens.map((t) => ({ text: t.text, color: t.role === "opt" ? C.ink : t.role === "val" ? C.edge : C.edge, bold: t.role === "opt", under: t.role === "bad" })),
    ];
    let x = x0;
    for (const w of words) {
      ctx.font = `${w.bold ? 700 : 400} ${fs}px JetBrains Mono, monospace`;
      const tw = ctx.measureText(w.text).width;
      if (x + tw > W - 10 && x > x0) { x = x0 + 18; y += lh; }
      ctx.fillStyle = w.color;
      ctx.fillText(w.text, x, y);
      if (w.under) { ctx.fillRect(x, y + 3, tw, 2); }
      x += tw + ctx.measureText(" ").width;
    }
    y += lh * 0.6;
    ctx.strokeStyle = C.faint;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(W - 10, y); ctx.stroke();
    y += lh;
    // 2) Namespace: 주어진 값(또는 parser 기본값)만
    ctx.font = `700 ${fs}px JetBrains Mono, monospace`;
    ctx.fillStyle = C.mid;
    ctx.fillText("parse_args →", x0, y);
    y += lh;
    ctx.font = `${fs}px JetBrains Mono, monospace`;
    if (r.ns) {
      const items = OPTS.filter((op) => op.dest !== "sets" || (r.ns!.sets as string[]).length)
        .map((op) => {
          const v = r.ns![op.dest];
          if (op.kind === "append") return `${op.dest}=[${(v as string[]).map((s) => `'${s}'`).join(", ")}]`;
          if (v === undefined) return null;
          if (op.kind === "flag" && v === false) return null;
          return `${op.dest}=${op.kind === "path" ? `'${v}'` : pyRepr(v as Val, r.floats.has(op.dest))}`;
        })
        .filter((s): s is string => s !== null);
      const nNone = OPTS.filter((op) => op.kind !== "flag" && op.kind !== "append" && r.ns![op.dest] === undefined).length;
      if (nNone) items.push(`나머지 ${nNone}개 None`);
      let xx = x0 + 12;
      for (const it of items) {
        const tw = ctx.measureText(it).width;
        if (xx + tw > W - 10 && xx > x0 + 12) { xx = x0 + 12; y += lh; }
        ctx.fillStyle = it.startsWith("나머지") ? C.mid : C.ink;
        ctx.fillText(it, xx, y);
        xx += tw + 16;
      }
    } else {
      ctx.fillStyle = C.mid;
      ctx.fillText("(Namespace 없음)", x0 + 12, y);
    }
    y += lh * 0.6;
    ctx.strokeStyle = C.faint;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(W - 10, y); ctx.stroke();
    y += lh;
    // 3) 병합 결과(dry-run 표) 또는 오류
    if (r.error) {
      ctx.font = `700 ${fs}px JetBrains Mono, monospace`;
      ctx.fillStyle = C.edge;
      for (const ln of wrap(r.error, W - x0 - 10)) { ctx.fillText(ln, x0, y); y += lh; }
      ctx.font = `${fs}px JetBrains Mono, monospace`;
      ctx.fillStyle = C.mid;
      ctx.fillText("종료 코드 2", x0, y);
      return;
    }
    ctx.font = `700 ${fs}px JetBrains Mono, monospace`;
    ctx.fillStyle = C.mid;
    ctx.fillText(r.dryRun ? "--dry-run →" : "최종 설정 (이대로 학습한다) →", x0, y);
    y += lh;
    const fileVals = r.ns && r.ns.config !== undefined ? FILES[String(r.ns.config)] ?? {} : {};
    const lines = dryLines(r);
    const keys = Object.keys(r.cfg!).sort();
    lines.forEach((ln, i) => {
      const k = keys[i];
      const src = r.sources[k];
      const clobbered = k in fileVals && src === "cli" && !given(r).has(k);
      ctx.font = `${src === "default" ? 400 : 700} ${fs}px JetBrains Mono, monospace`;
      ctx.fillStyle = clobbered ? C.edge : src === "default" ? C.mid : C.ink;
      ctx.fillText(ln, x0 + 12, y);
      if (clobbered) {
        const tw = ctx.measureText(ln).width;
        ctx.font = `${fs}px JetBrains Mono, monospace`;
        const note = `← 파일 값 ${pyRepr(fileVals[k], FLOATS.has(k))}`;
        if (x0 + 12 + tw + 10 + ctx.measureText(note).width < W) ctx.fillText(note, x0 + 12 + tw + 10, y);
      }
      y += lh;
    });
  }
  function wrap(s: string, w: number) {
    const out: string[] = [];
    let line = "";
    for (const word of s.split(" ")) {
      const t = line ? `${line} ${word}` : word;
      if (f.ctx.measureText(t).width > w && line) { out.push(line); line = word; } else line = t;
    }
    if (line) out.push(line);
    return out;
  }

  // 조작
  const pre = presets(PRESETS.map(({ key, label }) => ({ key, label })), (k) => { cur = PRESETS.find((p) => p.key === k)!; update(); }, cur.key);
  const sw = el("div", "wframe-row");
  const toggle = (label: string, on: boolean, fn: (v: boolean) => void) => {
    const t = el("label", "toggle");
    t.innerHTML = `<input type="checkbox" ${on ? "checked" : ""} /> `;
    t.append(document.createTextNode(label));
    t.querySelector("input")!.addEventListener("change", (e) => fn((e.target as HTMLInputElement).checked));
    sw.append(t);
  };
  toggle("parser 기본값 None", opts.noneDefaults, (v) => { opts.noneDefaults = v; update(); });
  toggle("allow_abbrev=False", opts.noAbbrev, (v) => { opts.noAbbrev = v; update(); });
  const ro = readouts();
  pre.row.style.flexWrap = "nowrap"; // 좁은 화면에서는 묶음 안에서 가로로 넘긴다(알약 모양 유지)
  f.panel.append(pre.row, sw, ro.row);

  function update() {
    const r = run(cur.argv.split(" "), opts);
    ro.set("exit", r.error ? "종료 코드 2" : "종료 코드 0", r.error ? "hot" : "ok");
    ro.set("stage", r.error ? (r.stage === "parse" ? "argparse에서 거부" : "설정 검증에서 거부") : r.dryRun ? "dry-run: 설정만 출력" : "학습 실행");
    if (r.cfg) {
      const n = (s: string) => Object.values(r.sources).filter((v) => v === s).length;
      ro.set("src", `출처 default ${n("default")} · file ${n("file")} · cli ${n("cli")}`);
      const file = r.ns && r.ns.config !== undefined ? FILES[String(r.ns.config)] ?? {} : {};
      const lost = Object.keys(file).filter((k) => r.sources[k] === "cli" && !given(r).has(k));
      ro.set("lost", `명령행에서 주지 않았는데 지워진 파일 값 ${lost.length}개`, lost.length ? "hot" : "");
    } else {
      ro.set("src", "최종 설정 없음");
      ro.set("lost", "—");
    }
    draw();
  }

  f.onResize(draw);
  update();
}
