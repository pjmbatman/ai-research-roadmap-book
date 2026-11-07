// 실습실(데스크톱, ADR 0012). 창: 안내 · 편집기 · 채점 · 터미널 · 파일. 넓은 화면은 겹치지 않게 바둑판으로 놓고
// 끌어 옮길 수 있다. 좁은 화면은 한 번에 창 하나를 전체 화면으로, 독은 아래 탭 막대가 된다.
// 학습자 코드는 submission.py로 두고 저장소의 test_exercise.py를 브라우저 Python에서 그대로 돌려 채점한다(harness.ts).
import { createEditor } from "./editor";
import { session, onRuntimeState, type RuntimeState } from "./runtime";
import { HARNESS } from "./harness";
import { progress } from "./store";
import { cleanError } from "./cells";
import { burst } from "./quest-state";
import { ICONS } from "../lib/icons";
import type { PageData } from "./main";

interface LabData { files: Record<string, string>; other: string[]; guide: string; dir: string; questUrl: string; title: string }
interface Ex { id: string; titleText: string; problem: string; starter: string; tests: string; solution: string; objective: number }

const HOME = "/home/pyodide/lab";
const HEAVY = /^\s*(import|from)\s+(matplotlib|torch|jax|tensorflow)\b/m;
const WINS = [
  { id: "guide", title: "안내", icon: ICONS.guide, c: "#8e7bff" },
  { id: "edit", title: "편집기", icon: ICONS.editor, c: "#0a84ff" },
  { id: "test", title: "채점", icon: ICONS.check, c: "#30d158", ic: "#062b12" },
  { id: "term", title: "터미널", icon: ICONS.terminal, c: "#2c2c2e" },
  { id: "file", title: "파일", icon: ICONS.folder, c: "#ff9f0a", ic: "#2a1a00" },
] as const;
type WinId = (typeof WINS)[number]["id"];
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const small = () => matchMedia("(max-width: 700px)").matches;
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export function mountLab(page: PageData) {
  const root = document.querySelector<HTMLElement>("[data-lab]")!;
  const lab: LabData = JSON.parse(document.getElementById("lab-data")!.textContent || "{}");
  const exs: Ex[] = Object.values(page.exercises);
  const ch = page.chapterId;
  const desk = root.querySelector<HTMLElement>("[data-desk]")!;
  const dock = root.querySelector<HTMLElement>("[data-dock]")!;
  const wins = {} as Record<WinId, HTMLElement>;
  const dockBtn = {} as Record<WinId, HTMLButtonElement>;
  let z = 10;

  /* ---------- 창 ---------- */
  for (const w of WINS) {
    const b = document.createElement("button");
    b.type = "button";
    b.style.setProperty("--c", w.c);
    if ("ic" in w) b.style.setProperty("--ic", w.ic);
    b.innerHTML = `${w.icon}<span class="tip">${w.title}</span>`;
    b.setAttribute("aria-label", w.title);
    b.addEventListener("click", () => { const el = wins[w.id]; if (el.hidden) openWin(w.id); else if (Number(el.style.zIndex) === z && !small()) { el.hidden = true; syncDock(); } else focusWin(el); });
    dock.append(b);
    dockBtn[w.id] = b;
    const el = document.createElement("section");
    el.className = "win";
    el.dataset.w = w.id;
    el.setAttribute("aria-label", w.title);
    el.innerHTML = `<div class="w-top"><span class="t">${w.title}</span><button type="button" class="mn" aria-label="최소화">–</button><button type="button" class="x" aria-label="닫기">×</button></div><div class="w-body"></div>`;
    desk.append(el);
    wins[w.id] = el;
    el.addEventListener("pointerdown", () => focusWin(el));
    el.querySelectorAll<HTMLButtonElement>(".mn, .x").forEach((x) => x.addEventListener("click", (e) => { e.stopPropagation(); el.hidden = true; syncDock(); }));
    drag(el);
  }
  const body = (id: WinId) => wins[id].querySelector<HTMLElement>(".w-body")!;
  function focusWin(el: HTMLElement) {
    if (small()) Object.values(wins).forEach((x) => { if (x !== el) x.hidden = true; });
    el.hidden = false;
    el.style.zIndex = String(++z);
    Object.values(wins).forEach((x) => x.classList.toggle("inactive", x !== el));
    syncDock();
    if (el.dataset.w === "term") setTimeout(() => tin.focus({ preventScroll: true }), 0);
  }
  function openWin(id: WinId) {
    const el = wins[id];
    const was = el.hidden;
    focusWin(el);
    if (was && !reduced()) { el.classList.remove("opening"); void el.offsetWidth; el.classList.add("opening"); }
  }
  const syncDock = () => WINS.forEach((w) => dockBtn[w.id].classList.toggle("open", !wins[w.id].hidden));
  function drag(el: HTMLElement) {
    const tb = el.querySelector<HTMLElement>(".w-top")!;
    let d: { x: number; y: number; l: number; t: number } | null = null;
    tb.addEventListener("pointerdown", (e) => { if ((e.target as HTMLElement).closest("button") || small()) return; d = { x: e.clientX, y: e.clientY, l: el.offsetLeft, t: el.offsetTop }; tb.setPointerCapture(e.pointerId); tb.style.cursor = "grabbing"; });
    tb.addEventListener("pointermove", (e) => { if (!d) return; el.style.left = `${clamp(d.l + e.clientX - d.x, 80 - el.offsetWidth, innerWidth - 80)}px`; el.style.top = `${clamp(d.t + e.clientY - d.y, 0, innerHeight - 90)}px`; el.dataset.moved = "1"; });
    tb.addEventListener("pointerup", () => { d = null; tb.style.cursor = ""; });
  }
  function layout(force = false) {
    if (small()) { Object.values(wins).forEach((w) => w.removeAttribute("style")); return; }
    const W = innerWidth, H = innerHeight - 38, L = 78, g = 12, top = 12;
    const pw = Math.round(clamp(W * 0.27, 300, 430)), tw = Math.round(clamp(W * 0.2, 250, 330)), h1 = Math.round(H * 0.6);
    const set = (id: WinId, x: number, y: number, w: number, h: number) => { const el = wins[id]; if (!force && el.dataset.moved) return; Object.assign(el.style, { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px`, zIndex: el.style.zIndex || "1" }); };
    set("guide", L, top, pw, h1);
    set("edit", L + pw + g, top, W - L - pw - tw - 3 * g, h1);
    set("test", W - tw - g, top, tw, h1);
    set("file", L, top + h1 + g, pw, H - h1 - top - 2 * g);
    set("term", L + pw + g, top + h1 + g, W - L - pw - 2 * g, H - h1 - top - 2 * g);
  }
  addEventListener("resize", () => layout());

  /* ---------- 안내 ---------- */
  const gb = body("guide");
  gb.innerHTML = `<div class="w-tabs" role="tablist"></div><div class="guide prose"></div>`;
  const gTabs = gb.querySelector<HTMLElement>(".w-tabs")!, gView = gb.querySelector<HTMLElement>(".guide")!;
  const guideItems = [{ key: "readme", label: "안내", html: lab.guide || "<p>이 장의 연습문제를 푼다.</p>" }, ...exs.map((e) => ({ key: e.id, label: e.id, html: `<h2>${esc(e.titleText)}</h2>${e.problem}` }))];
  guideItems.forEach((it) => { const b = document.createElement("button"); b.type = "button"; b.setAttribute("role", "tab"); b.dataset.k = it.key; b.textContent = it.label; b.addEventListener("click", () => showGuide(it.key, true)); gTabs.append(b); });
  function showGuide(k: string, follow = false) {
    const it = guideItems.find((x) => x.key === k) ?? guideItems[0];
    gView.innerHTML = it.html;
    gView.scrollTop = 0;
    gTabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.k === it.key)));
    if (follow && exs.some((e) => e.id === k)) openTab(`ex:${k}`);
  }

  /* ---------- 편집기 ---------- */
  const eb = body("edit");
  eb.innerHTML = `<div class="w-tabs" role="tablist"></div><div class="editors"></div><div class="ed-bar"><span class="ed-path"></span><button class="dbtn" type="button" data-act="reset">처음 코드로</button><button class="dbtn" type="button" data-act="solve">해답 보기</button><button class="dbtn run" type="button" data-act="run">▶ 실행</button><button class="dbtn chk" type="button" data-act="check">✓ 채점</button></div>`;
  const eTabs = eb.querySelector<HTMLElement>(".w-tabs")!, eHost = eb.querySelector<HTMLElement>(".editors")!, ePath = eb.querySelector<HTMLElement>(".ed-path")!;
  const editors = new Map<string, { host: HTMLElement; get: () => string; set: (s: string) => void; ro: boolean }>();
  const showingSolution = new Map<string, string | null>();
  let curTab = "";
  const exOfTab = (t: string) => exs.find((e) => t === `ex:${e.id}`);
  const subPath = (e: Ex) => `exercises/${e.id}/submission.py`;
  const codeOf = (e: Ex) => editors.get(`ex:${e.id}`)?.get() ?? progress.get(ch).exercises?.[e.id]?.code ?? e.starter;
  function openTab(t: string) {
    if (!editors.has(t)) {
      const host = document.createElement("div");
      eHost.append(host);
      const e = exOfTab(t);
      const file = t.startsWith("f:") ? t.slice(2) : "";
      let timer: number | undefined;
      const ed = createEditor(host, e ? codeOf(e) : lab.files[file] ?? "", {
        label: e ? `${e.titleText} 코드` : file, readOnly: !e,
        onRun: () => (e ? check(e.id) : undefined),
        onChange: e ? (code) => { if (showingSolution.get(e.id) != null) return; clearTimeout(timer); timer = window.setTimeout(() => progress.update(ch, (p) => { const prev = (p.exercises ??= {})[e.id]; p.exercises[e.id] = { passed: prev?.passed ?? false, code }; }), 400); } : undefined,
      });
      editors.set(t, { host, get: ed.get, set: ed.set, ro: !e });
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "tab"); b.dataset.t = t;
      b.textContent = e ? subPath(e).replace("exercises/", "") : file.split("/").slice(-2).join("/");
      b.title = e ? subPath(e) : file;
      b.addEventListener("click", () => openTab(t));
      eTabs.append(b);
    }
    curTab = t;
    editors.forEach((v, k) => (v.host.hidden = k !== t));
    eTabs.querySelectorAll<HTMLElement>("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.t === t)));
    eTabs.querySelector<HTMLElement>(`button[data-t="${CSS.escape(t)}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
    const e = exOfTab(t);
    ePath.textContent = `~/lab/${e ? subPath(e) : t.slice(2)}${e ? "" : " · 읽기 전용"}`;
    eb.querySelectorAll<HTMLButtonElement>('[data-act="reset"], [data-act="solve"], [data-act="check"]').forEach((b) => (b.disabled = !e));
    const sb = eb.querySelector<HTMLButtonElement>('[data-act="solve"]')!;
    sb.textContent = e && showingSolution.get(e.id) != null ? "내 코드로" : "해답 보기";
    if (e && !e.solution) sb.disabled = true;
    if (e) {
      gTabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.k === e.id)));
      gView.innerHTML = guideItems.find((x) => x.key === e.id)!.html;
    }
  }
  eb.querySelector('[data-act="reset"]')!.addEventListener("click", () => {
    const e = exOfTab(curTab); if (!e) return;
    const ed = editors.get(curTab)!;
    if (ed.get() !== e.starter && !confirm("작성한 코드를 지우고 처음 코드로 되돌릴까?")) return;
    showingSolution.set(e.id, null);
    ed.set(e.starter);
    openTab(curTab);
  });
  eb.querySelector('[data-act="solve"]')!.addEventListener("click", () => {
    const e = exOfTab(curTab); if (!e) return;
    const ed = editors.get(curTab)!;
    const mine = showingSolution.get(e.id);
    if (mine == null) { showingSolution.set(e.id, ed.get()); ed.set(e.solution); }
    else { showingSolution.set(e.id, null); ed.set(mine); }
    openTab(curTab);
  });
  eb.querySelector('[data-act="run"]')!.addEventListener("click", () => { openWin("term"); const e = exOfTab(curTab); run(`python ${e ? subPath(e) : curTab.slice(2)}`); });
  eb.querySelector('[data-act="check"]')!.addEventListener("click", () => { const e = exOfTab(curTab); if (e) check(e.id); });

  /* ---------- 채점 ---------- */
  const tb = body("test");
  tb.innerHTML = `<div class="tests">${exs.map((e) => `<div class="tgrp" data-ex="${e.id}"><h4><span>${esc(e.id)}</span><span class="sum"></span></h4><ul></ul></div>`).join("")}<div class="done" hidden>연습문제를 모두 통과했다. 퀘스트로 돌아가 통과 기준을 확인한다.<a href="${lab.questUrl}">퀘스트로 돌아가기</a></div></div>`;
  const testNames = (src: string) => [...src.matchAll(/^def (test_\w+)/gm)].map((m) => m[1]);
  function paintTests(id: string, res?: { name: string; ok: boolean; msg?: string }[], running = false) {
    const g = tb.querySelector<HTMLElement>(`.tgrp[data-ex="${id}"]`)!;
    const e = exs.find((x) => x.id === id)!;
    const ul = g.querySelector("ul")!;
    if (res) ul.innerHTML = res.map((r) => `<li class="${r.ok ? "pass" : "fail"}" title="${esc(r.msg ?? "")}"><i></i><span>${esc(r.name)}</span></li>${r.ok ? "" : `<div class="msg">${esc((r.msg ?? "").split("\n").slice(0, 3).join("\n"))}</div>`}`).join("");
    else if (!ul.children.length || running) ul.innerHTML = testNames(e.tests).map((n) => `<li class="${running ? "run" : ""}"><i></i><span>${n}</span></li>`).join("");
    const passed = !!progress.get(ch).exercises?.[id]?.passed;
    const sum = g.querySelector<HTMLElement>(".sum")!;
    sum.textContent = res ? `${res.filter((r) => r.ok).length}/${res.length}` : passed ? "통과" : "채점 전";
    sum.classList.toggle("ok", res ? res.every((r) => r.ok) : passed);
    tb.querySelector<HTMLElement>(".done")!.hidden = !exs.every((x) => progress.get(ch).exercises?.[x.id]?.passed);
  }
  exs.forEach((e) => paintTests(e.id));
  async function check(id: string) {
    const e = exs.find((x) => x.id === id);
    if (!e) return;
    openWin("test");
    paintTests(e.id, undefined, true);
    const code = codeOf(e);
    const dir = `/home/pyodide/exercises/${e.id}`;
    out(`$ pytest ${lab.dir}/exercises/${e.id}  (submission.py)`, "dim");
    const r = await session(`exercise:${e.id}`).run(`${HARNESS}\nrun_exercise_tests(${JSON.stringify(dir)})`, { files: { [`${dir}/submission.py`]: code, [`${dir}/test_exercise.py`]: e.tests } });
    if (!r.ok) { out(cleanError(r.error || "채점기를 실행하지 못했다."), "er"); paintTests(e.id, [{ name: "채점기 실행", ok: false, msg: cleanError(r.error || "") }]); return; }
    const res: { name: string; ok: boolean; msg?: string }[] = JSON.parse(String(r.value));
    const pass = res.filter((x) => x.ok).length;
    res.forEach((x) => out(`${x.ok ? "PASSED" : "FAILED"} ${x.name}${x.ok ? "" : `\n    ${(x.msg ?? "").split("\n")[0]}`}`, x.ok ? "ok" : "er"));
    const all = res.length > 0 && pass === res.length;
    out(all ? `=== ${pass} passed ===` : `=== ${res.length - pass} failed, ${pass} passed ===`, all ? "ok" : "warn");
    // 해답을 띄워 놓고 통과한 것은 통과로 치지 않는다
    const earned = all && showingSolution.get(e.id) == null;
    if (all && !earned) out("해답 코드로 통과했다. 내 코드로 돌아가 직접 통과해야 기록된다.", "warn");
    progress.update(ch, (p) => { const prev = (p.exercises ??= {})[e.id]; p.exercises[e.id] = { passed: earned || !!prev?.passed, code: showingSolution.get(e.id) ?? code }; });
    paintTests(e.id, res);
    if (earned && exs.every((x) => progress.get(ch).exercises?.[x.id]?.passed)) {
      out("연습문제를 모두 통과했다. 퀘스트로 돌아가 통과 기준을 확인한다.", "ok");
      const rr = wins.test.getBoundingClientRect();
      burst(rr.left + rr.width / 2, rr.top + 80, 44);
    }
  }

  /* ---------- 파일 ---------- */
  const fb = body("file");
  const allFiles = [...Object.keys(lab.files), ...lab.other, ...exs.map(subPath)].sort();
  const dirs = new Set<string>();
  allFiles.forEach((p) => { const a = p.split("/"); for (let i = 1; i < a.length; i++) dirs.add(a.slice(0, i).join("/")); });
  {
    let html = `<div class="files"><div>📁 ~/lab</div>`;
    const kids = (d: string) => [...new Set(allFiles.filter((p) => (d ? p.startsWith(d + "/") : true)).map((p) => (d ? p.slice(d.length + 1) : p).split("/")[0]))].sort((a, b) => Number(dirs.has(d ? `${d}/${a}` : a)) * -1 + Number(dirs.has(d ? `${d}/${b}` : b)) || a.localeCompare(b));
    const walk = (d: string, depth: number) => kids(d).forEach((k) => {
      const p = d ? `${d}/${k}` : k, pad = "&nbsp;".repeat(depth * 3);
      if (dirs.has(p)) { html += `<div>${pad}📁 ${esc(k)}</div>`; walk(p, depth + 1); }
      else html += `<div class="f" data-p="${esc(p)}">${pad}${k.endsWith("submission.py") ? "✏️" : /\.(png|jpe?g|svg|gif)$/.test(k) ? "🖼" : "📄"} ${esc(k)}</div>`;
    });
    walk("", 1);
    fb.innerHTML = html + "</div>";
  }
  fb.querySelectorAll<HTMLElement>(".f").forEach((d) => d.addEventListener("click", () => {
    fb.querySelectorAll(".f").forEach((x) => x.classList.toggle("sel", x === d));
    const p = d.dataset.p!;
    const e = exs.find((x) => subPath(x) === p);
    if (e) { openTab(`ex:${e.id}`); openWin("edit"); return; }
    if (!(p in lab.files)) { openWin("term"); run(`cat ${p}`); return; }
    openTab(`f:${p}`); openWin("edit");
  }));

  /* ---------- 터미널: 흉내 셸 + 브라우저 Python ---------- */
  const trm = body("term");
  trm.innerHTML = `<div class="term"><div class="ln dim">브라우저 안 실습 환경이다. 리눅스 커널이 아니라 흉내 셸이고, python 실행과 채점은 브라우저 Python(Pyodide)이 실제로 한다.\n도움말: help</div><div class="in"><span class="ps"></span><input aria-label="명령" autocomplete="off" autocapitalize="off" spellcheck="false"></div></div>`;
  const term = trm.querySelector<HTMLElement>(".term")!, tin = trm.querySelector<HTMLInputElement>("input")!, ps = trm.querySelector<HTMLElement>(".ps")!;
  let cwd = "";
  const hist: string[] = [];
  let hi = 0;
  const prompt = () => `<span class="pr">learner@ai4s-lab</span>:<span class="pd">~/lab${cwd ? "/" + esc(cwd) : ""}</span>$ `;
  ps.innerHTML = prompt();
  function out(s: string, cls = "", raw = false) { const l = document.createElement("div"); l.className = `ln ${cls}`; if (raw) l.innerHTML = s; else l.textContent = s; term.insertBefore(l, term.lastElementChild); term.scrollTop = term.scrollHeight; }
  term.addEventListener("click", () => tin.focus({ preventScroll: true }));
  const resolve = (p = ""): string | null => {
    const parts = (p.startsWith("~") ? p.replace(/^~\/?(lab\/?)?/, "") : cwd ? `${cwd}/${p}` : p).split("/");
    const st: string[] = [];
    for (const x of parts) { if (!x || x === ".") continue; if (x === "..") { if (!st.length) return null; st.pop(); } else st.push(x); }
    return st.join("/");
  };
  const isFile = (r: string) => allFiles.includes(r);
  const textOf = (r: string) => { const e = exs.find((x) => subPath(x) === r); return e ? codeOf(e) : lab.files[r]; };
  tin.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { const v = tin.value; tin.value = ""; run(v); }
    else if (e.key === "ArrowUp") { e.preventDefault(); if (hi > 0) tin.value = hist[--hi]; }
    else if (e.key === "ArrowDown") { e.preventDefault(); if (hi < hist.length) tin.value = hist[++hi] ?? ""; }
    else if (e.key === "Tab") {
      e.preventDefault();
      const parts = tin.value.split(" "), last = parts.pop() ?? "", pref = cwd ? `${cwd}/` : "";
      const names = new Set<string>();
      [...allFiles, ...dirs].forEach((p) => { if (!p.startsWith(pref)) return; const rest = p.slice(pref.length); if (rest.startsWith(last)) names.add(rest.slice(0, rest.indexOf("/", last.length) + 1 || rest.length)); });
      const c = [...names];
      if (c.length === 1) tin.value = [...parts, c[0]].join(" ");
      else if (c.length > 1) { let k = 0; while (c.every((x) => x[k] && x[k] === c[0][k])) k++; tin.value = [...parts, c[0].slice(0, k)].join(" "); out(c.sort().join("  "), "dim"); }
    } else if (e.key === "l" && e.ctrlKey) { e.preventDefault(); term.querySelectorAll(".ln").forEach((l) => l.remove()); }
  });
  async function python(args: string[]) {
    const sh = session("lab:shell");
    const files: Record<string, string> = {};
    // 무거운 라이브러리를 쓰는 파일은 넘기지 않는다(워커가 import를 보고 패키지를 불러오므로)
    Object.entries(lab.files).forEach(([p, t]) => { if (!HEAVY.test(t)) files[`${HOME}/${p}`] = t; });
    exs.forEach((e) => (files[`${HOME}/${subPath(e)}`] = codeOf(e)));
    const opts = { files, onStdout: (s: string) => out(s.replace(/\n$/, "")) };
    let code: string;
    if (args[0] === "-c") code = `import os\nos.chdir(${JSON.stringify(HOME + (cwd ? "/" + cwd : ""))})\n${args.slice(1).join(" ")}`;
    else {
      const r = resolve(args[0]);
      if (r == null || !isFile(r) || !r.endsWith(".py")) { out(`python: '${args[0]}' 파일을 열 수 없다`, "er"); return; }
      const src = textOf(r) ?? "";
      if (HEAVY.test(src)) {
        const cmd = /실행:\s*(.+)/.exec(src)?.[1]?.trim();
        out(`${r}는 브라우저에서 돌리지 않는 라이브러리(matplotlib·PyTorch 등)나 큰 계산을 쓴다.${cmd ? `\n로컬에서 실행한다: ${cmd}` : ""}`, "warn");
        return;
      }
      const full = `${HOME}/${r}`;
      code = `import os, sys, runpy\nos.chdir(${JSON.stringify(full.slice(0, full.lastIndexOf("/")))})\nsys.argv = [${JSON.stringify(full)}]\nrunpy.run_path(${JSON.stringify(full)}, run_name="__main__")\nNone`;
    }
    const res = await sh.run(code, opts);
    if (!res.ok) out(cleanError(res.error || ""), "er");
  }
  async function run(line: string) {
    out(prompt() + esc(line), "", true);
    const cmd = line.trim();
    if (!cmd) return;
    hist.push(cmd); hi = hist.length;
    const [c, ...args] = (cmd.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) ?? []).map((a) => a.replace(/^["']|["']$/g, ""));
    switch (c) {
      case "help": out("ls [경로] · cd 경로 · pwd · cat 파일 · python 파일 · python -c \"코드\"\ncheck [연습문제 id] · pytest [폴더] · git status · nvidia-smi · uname -a · whoami · clear\n↑↓ 이전 명령 · Tab 자동 완성 · 편집기에서 Ctrl+Enter = 채점"); break;
      case "pwd": out(`/home/learner/lab${cwd ? "/" + cwd : ""}`); break;
      case "whoami": out("learner"); break;
      case "echo": out(args.join(" ")); break;
      case "date": out(new Date().toString()); break;
      case "clear": term.querySelectorAll(".ln").forEach((l) => l.remove()); break;
      case "history": out(hist.map((h, i) => `${String(i + 1).padStart(4)}  ${h}`).join("\n")); break;
      case "uname": out("Linux ai4s-lab (브라우저 안 실습 환경 · Python은 Pyodide · WebAssembly)"); break;
      case "nvidia-smi": out("GPU를 찾을 수 없다. 이 장의 실습은 CPU로 충분하다.\n(GPU가 꼭 필요한 코드는 책에서 exec=\"gpu\"로 표시한다.)", "warn"); break;
      case "ls": {
        const r = resolve(args.find((a) => !a.startsWith("-")) ?? "");
        if (r == null || (r && !dirs.has(r) && !isFile(r))) { out(`ls: '${args[0] ?? ""}'에 접근할 수 없다`, "er"); break; }
        if (isFile(r)) { out(r.split("/").pop()!); break; }
        const ks = new Set<string>();
        allFiles.forEach((p) => { if (r ? p.startsWith(r + "/") : true) { const rest = r ? p.slice(r.length + 1) : p; const k = rest.split("/")[0]; ks.add(rest.includes("/") ? k + "/" : k); } });
        out([...ks].sort().map((k) => (k.endsWith("/") ? `<span class="pd">${esc(k)}</span>` : esc(k))).join("  "), "", true);
        break;
      }
      case "cd": { const r = resolve(args[0] ?? "~"); if (r == null || (r && !dirs.has(r))) { out(`cd: ${args[0]}: 그런 디렉터리가 없다`, "er"); break; } cwd = r; ps.innerHTML = prompt(); break; }
      case "cat": {
        if (!args[0]) { out("cat: 파일 이름이 필요하다", "er"); break; }
        const r = resolve(args[0]);
        if (r == null || !isFile(r)) { out(`cat: ${args[0]}: 그런 파일이 없다`, "er"); break; }
        const t = textOf(r);
        out(t ?? "(그림 같은 글이 아닌 파일이다. 로컬에서 연다.)", t == null ? "dim" : "");
        break;
      }
      case "git": {
        if (args[0] !== "status") { out("이 실습실에서는 git status만 흉내 낸다", "dim"); break; }
        const changed = exs.filter((e) => codeOf(e) !== e.starter);
        out(changed.length ? `커밋하도록 정하지 않은 변경 사항:\n${changed.map((e) => `\t수정함:   ${subPath(e)}`).join("\n")}` : "커밋할 사항 없음, 작업 폴더 깨끗함");
        break;
      }
      case "python": case "python3": if (!args[0]) out("대화형 python은 열지 않는다. python 파일 또는 python -c \"코드\"를 쓴다.", "dim"); else await python(args); break;
      case "check": case "pytest": {
        const want = args[0] ? resolve(args[0]) ?? args[0] : cwd;
        const hit = exs.filter((e) => want.includes(e.id));
        for (const e of hit.length ? hit : exs) await check(e.id);
        break;
      }
      default: out(`${c}: 명령을 찾을 수 없다. help로 목록을 본다.`, "er");
    }
  }

  /* ---------- 위 막대: 시계·Python 상태 ---------- */
  const clock = root.querySelector<HTMLElement>("[data-clock]");
  const tick = () => { if (clock) clock.textContent = new Date().toLocaleString("ko-KR", { month: "long", day: "numeric", weekday: "short", hour: "numeric", minute: "2-digit" }); };
  tick(); setInterval(tick, 20000);
  const st = root.querySelector<HTMLElement>("[data-py]")!;
  const LABEL: Record<RuntimeState, string> = { idle: "Python 대기", loading: "Python 불러오는 중", ready: "Python 준비됨", running: "실행 중", error: "Python 불러오기 실패" };
  onRuntimeState((s) => { st.dataset.state = s; st.querySelector("span")!.textContent = LABEL[s]; });

  /* ---------- 시작 ---------- */
  layout(true);
  showGuide(exs[0]?.id ?? "readme");
  if (exs[0]) openTab(`ex:${exs[0].id}`);
  if (small()) focusWin(wins.edit);
  else { Object.values(wins).forEach((w) => (w.hidden = false)); focusWin(wins.edit); }
  // 처음 채점이 빠르도록 브라우저 Python과 numpy를 미리 불러 둔다
  session("lab:shell").run("import numpy").then((r) => out(r.ok ? "[python] 준비됨: Pyodide · numpy" : `[python] 불러오지 못했다. 네트워크를 확인한 뒤 다시 시도한다.`, r.ok ? "ok" : "er"));
  const boot = root.querySelector<HTMLElement>("[data-boot]")!;
  const bar = boot.querySelector<HTMLElement>(".bar i")!, msg = boot.querySelector<HTMLElement>("p")!;
  const stepsB: [string, number][] = [["커널 시작", 22], ["작업 폴더 연결: ~/lab", 48], ["창 배치", 76], ["데스크톱 준비", 100]];
  let i = 0;
  const next = () => {
    if (i >= stepsB.length) { boot.classList.add("fade"); setTimeout(() => (boot.hidden = true), 520); return; }
    msg.textContent = stepsB[i][0]; bar.style.width = `${stepsB[i][1]}%`; i++;
    setTimeout(next, reduced() ? 20 : 260);
  };
  next();
}
