// 퀘스트(Clay, ADR 0012). 구간 탭 + 지그재그 길. 노드를 누르면 수업 시트가 뜬다.
// 읽기 노드: 절의 본문을 쪽으로 나눠 한 쪽씩 넘긴다(위젯·코드 셀·상자는 한 쪽에 하나). 퀴즈 노드: 문항 하나.
// 실습실 노드: 실습실 페이지로 간다. 통과 기준: 퀴즈·연습문제·자기 점검을 모아 보이고 '장 완료'로 마친다.
import { progress } from "./store";
import { nodeDone, burst } from "./quest-state";
import { renderQuestion, type Q } from "./quiz";
import { ICONS } from "../lib/icons";
import type { PageData } from "./main";

interface QNode { id: string; kind: string; name: string; section?: number; quiz?: string; objective?: number }
interface QData { nodes: QNode[]; groups: { name: string; nodes: string[] }[]; exIds: string[]; labUrl: string; nextUrl: string; nextLabel: string; closeUrl: string }

const KIND_LABEL: Record<string, string> = { goal: "시작", read: "읽기", widget: "직접 조작", code: "코드 실행", quiz: "개념 퀴즈", pitfall: "주의", ref: "더 읽기", boss: "마지막 관문", lab: "실습실" };
const Z = [0, 0.72, 0.12, -0.68, 0.04];
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const icon = (k: string) => (ICONS as Record<string, string>)[k] ?? ICONS.read;

export function mountQuest(page: PageData) {
  const root = document.querySelector<HTMLElement>("[data-quest]");
  const dataEl = document.getElementById("quest-data");
  if (!root || !dataEl) return;
  const data: QData = JSON.parse(dataEl.textContent || "{}");
  const ch = page.chapterId;
  const store = document.querySelector<HTMLElement>("[data-store]")!;
  const steps = [...store.querySelectorAll<HTMLElement>(":scope > section.step")];
  const path = root.querySelector<HTMLElement>("[data-path]")!;
  const trail = path.querySelector<SVGSVGElement>("svg.trail")!;
  const groupEls = [...path.querySelectorAll<HTMLElement>(".path-group")];
  const tabs = [...root.querySelectorAll<HTMLButtonElement>(".sec-tabs button")];
  const btn = new Map([...path.querySelectorAll<HTMLButtonElement>(".node")].map((b) => [b.dataset.node!, b]));
  const idx = new Map(data.nodes.map((n, i) => [n.id, i]));
  const groupOf = (id: string) => data.groups.findIndex((g) => g.nodes.includes(id));
  const states = () => { const done = data.nodes.map((n) => nodeDone(ch, n, data.exIds)); return { done, cur: done.findIndex((d) => !d) }; };
  let sec = 0;

  /* ---------- 길 그리기 ---------- */
  function layout() {
    const ids = data.groups[sec]?.nodes ?? [];
    const n = ids.length;
    if (!n) return;
    path.classList.add("laid");
    const w = path.clientWidth;
    const ns = btn.get(ids[0])!.offsetWidth || 80;
    const stacked = matchMedia("(max-width: 1180px)").matches;
    const avail = stacked ? 0 : Math.max(520, innerHeight - (path.getBoundingClientRect().top + scrollY) - 24);
    const gap = n > 1 ? Math.min(160, Math.max(ns + 56, (avail - 200) / (n - 1))) : 0;
    const need = 100 + gap * (n - 1) + 90;
    const top = 100 + Math.max(0, (avail - need) / 2);
    path.style.minHeight = `${Math.max(need, avail)}px`;
    const amp = Math.min(150, Math.max(0, w / 2 - 110));
    const pts = ids.map((_, i) => [w / 2 + Z[i % 5] * amp, top + i * gap] as const);
    ids.forEach((id, i) => { const b = btn.get(id)!; b.style.left = `${pts[i][0]}px`; b.style.top = `${pts[i][1]}px`; });
    const { done } = states();
    const seg = (i: number) => { const [x0, y0] = pts[i - 1], [x1, y1] = pts[i], my = (y0 + y1) / 2; return `M${x0} ${y0} C${x0} ${my} ${x1} ${my} ${x1} ${y1}`; };
    let all = "", ok = "";
    for (let i = 1; i < n; i++) { all += seg(i); if (done[idx.get(ids[i - 1])!] && done[idx.get(ids[i])!]) ok += seg(i); }
    trail.innerHTML = `<path d="${all}" fill="none" stroke="var(--clay)" stroke-width="12" stroke-linecap="round" stroke-dasharray="0.1 24"/>` + (ok ? `<path d="${ok}" fill="none" stroke="var(--green)" stroke-width="12" stroke-linecap="round" stroke-dasharray="0.1 24"/>` : "");
  }

  function paint() {
    const { done, cur } = states();
    const any = done.some(Boolean);
    data.nodes.forEach((n, i) => {
      const b = btn.get(n.id)!;
      b.classList.toggle("done", done[i]);
      b.classList.toggle("cur", i === cur);
      b.querySelector(".ico")!.innerHTML = done[i] ? ICONS.done : icon(n.kind);
      b.querySelector(".tag")?.remove();
      if (i === cur) {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = n.kind === "lab" ? "실습실 입장" : n.kind === "boss" ? "마지막 관문" : !any ? "시작" : "다음";
        b.append(t);
      }
      b.setAttribute("aria-label", `${n.name}${done[i] ? " · 마침" : i === cur ? " · 지금 할 단계" : ""}`);
    });
    tabs.forEach((t, gi) => t.classList.toggle("done", data.groups[gi].nodes.every((id) => done[idx.get(id)!])));
    const count = done.filter(Boolean).length;
    const p = progress.get(ch);
    const quizOk = page.quiz.filter((q: Q) => p.quiz?.[q.id]?.correct).length;
    const exOk = data.exIds.filter((id) => p.exercises?.[id]?.passed).length;
    const set = (sel: string, v: string) => { const e = document.querySelector<HTMLElement>(sel); if (e) e.textContent = v; };
    set("[data-q-count]", `${count} / ${data.nodes.length}`);
    set('[data-q-k="steps"]', `${count}/${data.nodes.length}`);
    set('[data-q-k="quiz"]', `${quizOk}/${page.quiz.length}`);
    set('[data-q-k="ex"]', `${exOk}/${data.exIds.length}`);
    const meter = document.querySelector<HTMLElement>("[data-q-meter]");
    if (meter) { meter.style.width = `${(100 * count) / Math.max(1, data.nodes.length)}%`; meter.parentElement?.setAttribute("aria-valuenow", String(Math.round((100 * count) / Math.max(1, data.nodes.length)))); }
    // 학습 목표: 그 목표에 걸린 퀴즈·연습문제를 모두 마치면 체크(걸린 것이 없으면 장을 마쳤을 때)
    document.querySelectorAll<HTMLElement>("[data-obj]").forEach((li) => {
      const k = Number(li.dataset.obj);
      const qs = page.quiz.filter((q: Q) => q.objective === k);
      const ex = Object.values(page.exercises).filter((e: any) => e.objective === k);
      const ok = qs.length + ex.length > 0 ? qs.every((q: Q) => p.quiz?.[q.id]?.correct) && ex.every((e: any) => p.exercises?.[e.id]?.passed) : done[done.length - 1];
      li.classList.toggle("ok", !!ok);
      li.querySelector("i")!.textContent = ok ? "✓" : String(k);
    });
    layout();
  }

  function showGroup(i: number, dir = 1) {
    if (i === sec || i < 0 || i >= groupEls.length) return;
    groupEls[sec].hidden = true;
    sec = i;
    const g = groupEls[i];
    g.hidden = false;
    tabs.forEach((t, k) => t.setAttribute("aria-selected", String(k === i)));
    tabs[i]?.scrollIntoView({ block: "nearest", inline: "center" });
    if (!reduced()) { g.classList.remove("in", "back"); trail.style.opacity = "0"; void g.offsetWidth; g.classList.add("in"); if (dir < 0) g.classList.add("back"); setTimeout(() => (trail.style.opacity = "1"), 220); }
    layout();
  }
  tabs.forEach((t, i) => t.addEventListener("click", () => showGroup(i, i > sec ? 1 : -1)));
  btn.forEach((b, id) => b.addEventListener("click", () => open(data.nodes[idx.get(id)!], b)));
  addEventListener("resize", () => layout());
  window.addEventListener("progress-change", paint);

  /* ---------- 수업 시트 ---------- */
  let sheet: { el: HTMLElement; close: () => void; key: (e: KeyboardEvent) => void } | null = null;

  function shell(n: QNode, segs = 1) {
    const bg = document.createElement("div");
    bg.className = "ls-bg";
    bg.innerHTML = `<div class="ls" role="dialog" aria-modal="true"><div class="ls-top"><button class="ls-x" type="button" aria-label="닫기">${ICONS.close}</button><div class="ls-prog">${"<i></i>".repeat(Math.max(1, segs))}</div><span class="ls-k"></span></div><div class="ls-name"></div><div class="ls-body"></div><div class="ls-foot"><button class="cbtn ghost" type="button" data-prev>이전</button><div class="ls-msg" aria-live="polite"></div><button class="cbtn" type="button" data-next>계속</button></div></div>`;
    bg.querySelector(".ls")!.setAttribute("aria-label", n.name);
    bg.querySelector(".ls-name")!.textContent = `${KIND_LABEL[n.kind] ?? "읽기"} · ${n.name}`;
    document.body.append(bg);
    document.documentElement.style.overflow = "hidden";
    requestAnimationFrame(() => bg.classList.add("on"));
    return {
      bg,
      body: bg.querySelector<HTMLElement>(".ls-body")!,
      foot: bg.querySelector<HTMLElement>(".ls-foot")!,
      msg: bg.querySelector<HTMLElement>(".ls-msg")!,
      prev: bg.querySelector<HTMLButtonElement>("[data-prev]")!,
      next: bg.querySelector<HTMLButtonElement>("[data-next]")!,
      k: bg.querySelector<HTMLElement>(".ls-k")!,
      prog: [...bg.querySelectorAll<HTMLElement>(".ls-prog i")],
      x: bg.querySelector<HTMLButtonElement>(".ls-x")!,
    };
  }

  function open(n: QNode, from?: HTMLElement) {
    if (sheet) sheet.close();
    if (n.kind === "lab") { toLab(from); return; }
    const opener = from ?? btn.get(n.id) ?? null;
    let cleanup = () => {};
    let onKey: (e: KeyboardEvent) => void = () => {};
    const ui = shell(n);
    const close = () => {
      if (!sheet) return;
      cleanup();
      document.removeEventListener("keydown", onKeyWrap);
      ui.bg.classList.remove("on");
      document.documentElement.style.overflow = "";
      setTimeout(() => ui.bg.remove(), 240);
      sheet = null;
      if (location.hash) history.replaceState(null, "", location.pathname + location.search);
      (opener as HTMLElement | null)?.focus?.({ preventScroll: true });
    };
    const onKeyWrap = (e: KeyboardEvent) => onKey(e);
    document.addEventListener("keydown", onKeyWrap);
    sheet = { el: ui.bg, close, key: onKeyWrap };
    ui.x.addEventListener("click", close);
    ui.bg.addEventListener("click", (e) => { if (e.target === ui.bg) close(); });
    const finish = () => { complete(n); close(); };

    if (n.kind === "quiz") {
      const q = page.quiz.find((x: Q) => x.id === n.quiz) as Q | undefined;
      ui.foot.hidden = true;
      ui.k.textContent = "";
      if (!q) { ui.body.textContent = "문항을 찾을 수 없다."; return; }
      ui.bg.querySelector(".ls-name")!.textContent = `개념 퀴즈 · 학습 목표 ${q.objective}${q.objectiveText ? ` — ${q.objectiveText}` : ""}`;
      const box = renderQuestion(ch, q, { onContinue: finish, continueLabel: "계속" });
      box.classList.add("ls-page");
      ui.body.append(box);
      onKey = (e) => { if (e.key === "Escape") close(); };
      requestAnimationFrame(() => box.querySelector<HTMLElement>(".quiz-choice, input")?.focus({ preventScroll: true }));
      return;
    }

    // 읽기·통과 기준: 절 본문을 시트로 옮겨 보이고, 닫을 때 제자리로 돌려놓는다
    const step = n.section != null ? steps[n.section] : undefined;
    if (!step) { ui.body.textContent = "본문을 찾을 수 없다."; return; }
    const holder = document.createComment("step");
    step.before(holder);
    ui.body.append(step);
    const h2 = step.querySelector("h2");
    if (h2?.id) history.replaceState(null, "", `#${h2.id}`);
    const blocks = blocksOf(step);
    cleanup = () => { blocks.forEach((b) => b.classList.remove("pg-off")); step.classList.remove("ls-page", "back"); holder.replaceWith(step); };

    if (n.kind === "boss") { const off = bossView(n, ui, step, close); const restore = cleanup; cleanup = () => { off(); restore(); }; onKey = (e) => { if (e.key === "Escape") close(); }; return; }

    const pages = paginate(blocks);
    ui.bg.querySelector(".ls-prog")!.innerHTML = "<i></i>".repeat(pages.length);
    ui.prog.splice(0, ui.prog.length, ...ui.bg.querySelectorAll<HTMLElement>(".ls-prog i"));
    let k = 0;
    const show = (i: number, dir = 1) => {
      k = Math.max(0, Math.min(pages.length - 1, i));
      const on = new Set(pages[k]);
      blocks.forEach((b) => b.classList.toggle("pg-off", !on.has(b)));
      if (!reduced()) { step.classList.remove("ls-page", "back"); void step.offsetWidth; step.classList.add("ls-page"); if (dir < 0) step.classList.add("back"); }
      ui.body.scrollTop = 0;
      ui.prog.forEach((p, j) => p.classList.toggle("on", j < k || (j === k && j === pages.length - 1 && nodeDone(ch, n, data.exIds))));
      ui.k.textContent = `${k + 1} / ${pages.length}`;
      ui.prev.hidden = k === 0;
      const last = k === pages.length - 1;
      ui.next.textContent = last ? "완료" : "계속";
      ui.next.className = `cbtn${last ? " green" : ""}`;
      const has = (sel: string) => pages[k].some((b) => b.matches(sel) || b.querySelector(sel));
      ui.msg.textContent = has(".widget") ? "직접 움직여 보고 넘어간다." : has('.cell[data-exec="browser"]') ? "▶ 실행으로 직접 돌려 볼 수 있다." : "";
      // 이 쪽의 위젯·편집기가 새 크기를 잡도록
      window.dispatchEvent(new Event("resize"));
    };
    ui.prev.addEventListener("click", () => show(k - 1, -1));
    ui.next.addEventListener("click", () => { ui.prog[k]?.classList.add("on"); if (k < pages.length - 1) show(k + 1); else finish(); });
    onKey = (e) => {
      const t = e.target as HTMLElement;
      if (e.key === "Escape") { close(); return; }
      if (e.metaKey || e.ctrlKey || e.altKey || t.closest("input, textarea, select, .cm-editor, [contenteditable], canvas")) return;
      if (e.key === "ArrowRight") { e.preventDefault(); if (k < pages.length - 1) show(k + 1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); show(k - 1, -1); }
      else if (e.key === "Enter" && !t.closest("button, a")) { e.preventDefault(); ui.next.click(); }
    };
    show(0);
    requestAnimationFrame(() => ui.next.focus({ preventScroll: true }));
  }

  function bossView(n: QNode, ui: ReturnType<typeof shell>, step: HTMLElement, close: () => void): () => void {
    ui.k.textContent = "";
    ui.prev.hidden = true;
    const list = document.createElement("div");
    list.className = "boss-list";
    step.prepend(list);
    const checks = () => [...step.querySelectorAll<HTMLInputElement>(".box-pass input.check")];
    const render = () => {
      const p = progress.get(ch);
      const qOk = page.quiz.filter((q: Q) => p.quiz?.[q.id]?.correct).length;
      const eOk = data.exIds.filter((id) => p.exercises?.[id]?.passed).length;
      const cOk = checks().filter((c) => c.checked).length;
      const rows = [
        page.quiz.length ? { ok: qOk === page.quiz.length, t: "개념 퀴즈를 모두 맞힌다", v: `${qOk}/${page.quiz.length}` } : null,
        data.exIds.length ? { ok: eOk === data.exIds.length, t: "연습문제 테스트를 모두 통과한다", v: `${eOk}/${data.exIds.length}` } : null,
        checks().length ? { ok: cOk === checks().length, t: "아래 기준을 스스로 확인해 체크한다", v: `${cOk}/${checks().length}` } : null,
      ].filter(Boolean) as { ok: boolean; t: string; v: string }[];
      list.innerHTML = rows.map((r, i) => `<div class="boss-row${r.ok ? " ok" : ""}"><span class="k">${r.ok ? "✓" : i + 1}</span><span>${r.t}</span><b>${r.v}</b></div>`).join("");
      const finished = nodeDone(ch, n, data.exIds);
      ui.foot.className = "ls-foot";
      ui.next.onclick = null;
      if (finished) {
        ui.foot.classList.add("good");
        ui.msg.textContent = "이 장을 마쳤다.";
        ui.next.className = "cbtn green";
        ui.next.textContent = data.nextUrl ? "다음 장 →" : "닫기";
        ui.next.onclick = () => (data.nextUrl ? (location.href = data.nextUrl) : close());
      } else if (qOk < page.quiz.length) {
        const miss = data.nodes.find((x) => x.kind === "quiz" && !p.quiz?.[x.quiz!]?.correct);
        ui.msg.textContent = `맞히지 못한 퀴즈가 ${page.quiz.length - qOk}개 남았다.`;
        ui.next.className = "cbtn";
        ui.next.textContent = "퀴즈로";
        ui.next.onclick = () => { close(); if (miss) { showGroup(groupOf(miss.id)); setTimeout(() => open(miss), 260); } };
      } else if (eOk < data.exIds.length) {
        ui.msg.textContent = `연습문제 ${data.exIds.length - eOk}개가 남았다.`;
        ui.next.className = "cbtn orange";
        ui.next.textContent = "실습실로";
        ui.next.onclick = () => { close(); toLab(btn.get(n.id)); };
      } else if (cOk < checks().length) {
        ui.msg.textContent = "아래 기준을 읽고 스스로 확인한 것에 체크한다.";
        ui.next.className = "cbtn";
        ui.next.textContent = "장 완료";
        ui.next.disabled = true;
        return;
      } else {
        ui.msg.textContent = "모든 기준을 채웠다.";
        ui.next.className = "cbtn green";
        ui.next.textContent = "장 완료";
        ui.next.onclick = () => { complete(n); const r = ui.next.getBoundingClientRect(); burst(r.left + r.width / 2, r.top, 48); render(); };
      }
      ui.next.disabled = false;
    };
    const onChange = () => render();
    step.addEventListener("change", onChange);
    window.addEventListener("progress-change", onChange);
    render();
    return () => { list.remove(); step.removeEventListener("change", onChange); window.removeEventListener("progress-change", onChange); };
  }

  function complete(n: QNode) {
    const was = nodeDone(ch, n, data.exIds);
    if (n.kind !== "quiz") progress.update(ch, (p) => { (p.steps ??= {})[n.id] = true; });
    const b = btn.get(n.id);
    if (!was && b && nodeDone(ch, n, data.exIds)) {
      const r = b.getBoundingClientRect();
      setTimeout(() => burst(r.left + r.width / 2, r.top + r.height / 2), 200);
      const gi = groupOf(n.id);
      const st = states();
      if (data.groups[gi].nodes.every((id) => st.done[idx.get(id)!]) && gi + 1 < data.groups.length) setTimeout(() => showGroup(gi + 1), 900);
    }
  }

  function toLab(from?: HTMLElement | null) {
    if (!data.labUrl) return;
    if (reduced() || !from) { location.href = data.labUrl; return; }
    const r = from.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const R = Math.hypot(Math.max(cx, innerWidth - cx), Math.max(cy, innerHeight - cy));
    const d = document.createElement("div");
    d.className = "ripple";
    d.style.cssText = `left:${cx - R}px;top:${cy - R}px;width:${R * 2}px;height:${R * 2}px`;
    document.body.append(d);
    d.animate([{ transform: "scale(0)" }, { transform: "scale(1)" }], { duration: 520, easing: "cubic-bezier(.6,0,.4,1)", fill: "forwards" }).finished.then(() => { location.href = data.labUrl; });
  }

  /* ---------- 시작 ---------- */
  const { cur } = states();
  const fromHash = () => {
    const id = decodeURIComponent(location.hash.slice(1));
    const target = id ? document.getElementById(id) : null;
    if (!target || !store.contains(target)) return false;
    const si = steps.findIndex((s) => s.contains(target));
    const node = data.nodes.find((x) => x.section === si && x.kind !== "quiz");
    if (!node || node.kind === "lab") return false;
    const gi = groupOf(node.id);
    if (gi !== sec) { groupEls[sec].hidden = true; sec = gi; groupEls[gi].hidden = false; tabs.forEach((t, k) => t.setAttribute("aria-selected", String(k === gi))); }
    paint();
    open(node);
    return true;
  };
  const startGroup = cur < 0 ? data.groups.length - 1 : groupOf(data.nodes[cur].id);
  if (startGroup > 0) { groupEls[0].hidden = true; sec = startGroup; groupEls[sec].hidden = false; tabs.forEach((t, k) => t.setAttribute("aria-selected", String(k === sec))); }
  paint();
  if (!fromHash()) tabs[sec]?.scrollIntoView({ block: "nearest", inline: "center" });
  window.addEventListener("hashchange", () => { if (!sheet) fromHash(); });
}

/** 절 본문을 쪽 나눔 단위로 편다: 도입 문단·개념 절은 자식 하나하나, 고정 절 상자는 통째로 */
function blocksOf(step: HTMLElement): HTMLElement[] {
  const out: HTMLElement[] = [];
  for (const c of [...step.children] as HTMLElement[]) {
    if (c.matches("div.lead")) out.push(...([...c.children] as HTMLElement[]));
    else if (c.matches("section.box")) out.push(c);
    else if (c.matches("section")) out.push(...([...c.children] as HTMLElement[]));
    else out.push(c);
  }
  return out.filter((b) => !b.matches(".quiz-slot, .boss-list"));
}

/** 쪽 나누기: 글은 한 쪽에 700자 안팎, 위젯·코드 셀·상자·표·그림은 한 쪽에 하나(바로 앞의 짧은 소개 문단과 함께). */
function paginate(blocks: HTMLElement[]): HTMLElement[][] {
  const heavy = (b: HTMLElement) => b.matches(".widget, .cell, aside.callout, table, pre, .box, figure") || !!b.querySelector(":scope > img, .widget, .cell");
  const weight = (b: HTMLElement) => (b.tagName === "H2" ? 0 : b.querySelector(".katex-display") ? 160 : (b.textContent || "").length);
  const pages: HTMLElement[][] = [];
  let cur: HTMLElement[] = [], len = 0, hasHeavy = false;
  const flush = () => { if (cur.some((b) => b.tagName !== "H2")) { pages.push(cur); cur = []; len = 0; hasHeavy = false; } };
  for (const b of blocks) {
    if (b.tagName === "H3") { flush(); cur.push(b); continue; }
    if (heavy(b)) {
      if (hasHeavy || len > 420) {
        // 바로 앞의 짧은 문단은 이 블록의 소개로 함께 넘긴다
        const last = cur[cur.length - 1];
        const carry = last && last.tagName === "P" && weight(last) < 200 && cur.length > 1 ? cur.pop()! : null;
        flush();
        if (carry) { cur.push(carry); len = weight(carry); }
      }
      cur.push(b); hasHeavy = true; len += 160;
      continue;
    }
    const w = weight(b);
    if (cur.length && (hasHeavy ? len + w > 520 : len + w > 760)) flush();
    cur.push(b); len += w;
  }
  flush();
  if (!pages.length) pages.push(blocks);
  // H2만 있는 첫 쪽이 생기지 않도록 앞쪽 H2를 첫 쪽에 붙인다
  const lone = blocks.filter((b) => b.tagName === "H2" && !pages.some((p) => p.includes(b)));
  if (lone.length) pages[0].unshift(...lone);
  return pages;
}
