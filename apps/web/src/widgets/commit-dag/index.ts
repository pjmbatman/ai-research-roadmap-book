// merge·rebase·reset과 커밋 그래프 (data/widgets.toml: commit-dag).
// 4.1장의 DAG(A ← B ← C ← F = main, B ← D ← E = feature)로 명령을 한 단계씩 실행하며 commit과 이름표(branch·HEAD)가 어떻게 바뀌는지 본다.
// commit은 parent를 가리키고(화살표는 과거로), branch는 commit 하나를 가리키는 움직이는 이름표다.
// 어떤 이름표에서도 도달할 수 없는 commit은 지워지지 않고 흐리게 남는다(rebase 전의 D·E, reset --hard 뒤의 B·C).
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

export interface Cm { id: string; p: string[]; x: number; y: number }
export interface St { cs: Cm[]; refs: Record<string, string>; head: string; cmd: string; reflog?: string[]; mark?: number }
type Step = (s: St) => void;

const clone = (s: St): St => ({ cs: s.cs.map((c) => ({ ...c, p: [...c.p] })), refs: { ...s.refs }, head: s.head, cmd: s.cmd, reflog: s.reflog ? [...s.reflog] : undefined });
const tip = (s: St) => s.refs[s.head];
const commit = (id: string, x: number, y: number, msg?: string): Step => (s) => {
  s.cs.push({ id, p: [tip(s)], x, y });
  s.refs[s.head] = id;
  s.cmd = "git commit";
  if (s.reflog && msg) s.reflog.unshift(`${id} ${s.head}@{0}: commit: ${msg}`);
};
const renumber = (s: St) => { if (s.reflog) s.reflog = s.reflog.map((l, i) => l.replace(/@\{\d+\}/, `@{${i}}`)); };

// 본문 그림의 DAG
const DAG: St = {
  cs: [{ id: "A", p: [], x: 0, y: 0 }, { id: "B", p: ["A"], x: 1, y: 0 }, { id: "C", p: ["B"], x: 2, y: 0 }, { id: "F", p: ["C"], x: 3, y: 0 }, { id: "D", p: ["B"], x: 2, y: 1 }, { id: "E", p: ["D"], x: 3, y: 1 }],
  refs: { main: "F", feature: "E" }, head: "main", cmd: "시작",
};
const MSG = { A: "model: baseline MLP", B: "train: 학습 루프", C: "eval: 평가 스크립트" };

export const SCENARIOS: Record<string, { label: string; start: St; steps: Step[] }> = {
  branch: {
    label: "commit·branch",
    start: { cs: [{ id: "A", p: [], x: 0, y: 0 }, { id: "B", p: ["A"], x: 1, y: 0 }], refs: { main: "B" }, head: "main", cmd: "시작" },
    steps: [
      (s) => { s.refs.feature = tip(s); s.head = "feature"; s.cmd = "git switch -c feature"; },
      commit("D", 2, 1), commit("E", 3, 1),
      (s) => { s.head = "main"; s.cmd = "git switch main"; },
      commit("C", 2, 0), commit("F", 3, 0),
    ],
  },
  merge: {
    label: "merge",
    start: DAG,
    steps: [
      (s) => { s.cs.push({ id: "M", p: ["F", "E"], x: 4, y: 0 }); s.refs.main = "M"; s.cmd = "git merge feature"; },
      (s) => { s.head = "feature"; s.cmd = "git switch feature"; },
      (s) => { s.refs.feature = "M"; s.cmd = "git merge main  (fast-forward)"; },
    ],
  },
  rebase: {
    label: "rebase",
    start: { ...DAG, head: "feature" },
    steps: [
      (s) => { s.cs.push({ id: "D′", p: ["F"], x: 4, y: 0 }, { id: "E′", p: ["D′"], x: 5, y: 0 }); s.refs.feature = "E′"; s.cmd = "git rebase main"; },
      (s) => { s.head = "main"; s.cmd = "git switch main"; },
      (s) => { s.refs.main = "E′"; s.cmd = "git merge feature  (fast-forward)"; },
    ],
  },
  reset: {
    label: "reset --hard",
    start: { cs: [{ id: "A", p: [], x: 0, y: 0 }], refs: { main: "A" }, head: "main", cmd: "시작", reflog: [`A main@{0}: commit: ${MSG.A}`] },
    steps: [
      (s) => { commit("B", 1, 0, MSG.B)(s); renumber(s); },
      (s) => { commit("C", 2, 0, MSG.C)(s); renumber(s); },
      (s) => { s.refs.main = "A"; s.cmd = "git reset --hard HEAD~2"; s.reflog!.unshift("A main@{0}: reset: moving to HEAD~2"); renumber(s); },
    ],
  },
  reflog: {
    label: "reflog 복구",
    start: { cs: [{ id: "A", p: [], x: 0, y: 0 }, { id: "B", p: ["A"], x: 1, y: 0 }, { id: "C", p: ["B"], x: 2, y: 0 }], refs: { main: "A" }, head: "main", cmd: "시작",
      reflog: ["A main@{0}: reset: moving to HEAD~2", `C main@{1}: commit: ${MSG.C}`, `B main@{2}: commit: ${MSG.B}`, `A main@{3}: commit: ${MSG.A}`] },
    steps: [
      (s) => { s.cmd = "git reflog main"; s.mark = 1; },
      (s) => { s.refs.rescue = "C"; s.cmd = "git branch rescue main@{1}"; s.mark = 1; },
      (s) => { s.refs.main = "C"; s.cmd = "git reset --hard rescue"; s.reflog!.unshift("C main@{0}: reset: moving to rescue"); renumber(s); s.mark = 0; },
    ],
  },
};

/** 시나리오의 단계별 상태 */
export function states(key: string): St[] {
  const sc = SCENARIOS[key];
  const out = [clone(sc.start)];
  for (const st of sc.steps) { const s = clone(out[out.length - 1]); s.mark = undefined; st(s); out.push(s); }
  return out;
}
/** 이름표(branch·HEAD)에서 parent를 따라 도달할 수 있는 commit */
export function reachable(s: St) {
  const par = new Map(s.cs.map((c) => [c.id, c.p]));
  const seen = new Set<string>(), stack = Object.values(s.refs);
  while (stack.length) { const x = stack.pop()!; if (!seen.has(x)) { seen.add(x); stack.push(...(par.get(x) ?? [])); } }
  return seen;
}

export function mount(root: HTMLElement) {
  const f = frame(root, "merge·rebase·reset과 커밋 그래프", "step", ((w) => (w < 400 ? 0.95 : w < 560 ? 0.72 : 0.44))(root.clientWidth || 600), "commit을 원, parent를 가리키는 화살표, branch와 HEAD를 이름표로 그린 커밋 그래프");
  let key = "rebase", S = states(key), k = 0;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const s = S[k], prev = S[Math.max(0, k - 1)];
    const reach = reachable(s);
    const newIds = new Set(k > 0 ? s.cs.filter((c) => !prev.cs.some((q) => q.id === c.id)).map((c) => c.id) : []);
    const movedRefs = new Set(k > 0 ? Object.keys(s.refs).filter((r) => prev.refs[r] !== s.refs[r]) : []);
    const headMoved = k > 0 && (prev.head !== s.head || prev.refs[prev.head] !== s.refs[s.head]);
    const narrow = W < 560;
    const hasLog = !!S[0].reflog;
    const pad = 16;
    // 그래프 영역
    const gx0 = pad, gx1 = hasLog && !narrow ? W * 0.56 : W - pad;
    const gy0 = 44, gy1 = hasLog && narrow ? H * 0.56 : H - 14;
    const cols = Math.max(...S[S.length - 1].cs.map((c) => c.x), ...S.flatMap((t) => t.cs.map((c) => c.x))) + 1;
    const rows = Math.max(...S.flatMap((t) => t.cs.map((c) => c.y))) + 1;
    const dx = Math.min(96, (gx1 - gx0 - 30) / Math.max(cols - 1, 1));
    const offX = gx0 + (gx1 - gx0 - dx * (cols - 1)) / 2;
    const rowGap = Math.min(110, (gy1 - gy0) / (rows + 0.6));
    const R = Math.min(17, dx * 0.24);
    const midY = Math.max(rows === 1 ? (gy0 + gy1) / 2 : gy0 + (gy1 - gy0 - rowGap) / 2, gy0 + R + 14 + 2 * 23 + 4);
    const P = (c: Cm): [number, number] => [offX + c.x * dx, midY + c.y * rowGap];
    const byId = new Map(s.cs.map((c) => [c.id, c]));

    // 명령
    ctx.textBaseline = "middle";
    ctx.font = "600 13px JetBrains Mono, monospace";
    ctx.fillStyle = k > 0 ? C.edge : C.mid;
    ctx.fillText(k > 0 ? `$ ${s.cmd}` : "시작 상태", pad, 20);

    // 간선: 자식 → parent (화살표 끝이 parent)
    for (const c of s.cs) {
      for (const pid of c.p) {
        const p = byId.get(pid)!;
        const [x1, y1] = P(c), [x0, y0] = P(p);
        const live = reach.has(c.id);
        const ang = Math.atan2(y0 - y1, x0 - x1);
        const sx = x1 + R * Math.cos(ang), sy = y1 + R * Math.sin(ang);
        const ex = x0 - (R + 2) * Math.cos(ang), ey = y0 - (R + 2) * Math.sin(ang);
        ctx.strokeStyle = newIds.has(c.id) ? C.edge : live ? C.mid : C.grid2;
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = 1.6;
        ctx.setLineDash(live ? [] : [4, 4]);
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - 8 * Math.cos(ang - 0.45), ey - 8 * Math.sin(ang - 0.45));
        ctx.lineTo(ex - 8 * Math.cos(ang + 0.45), ey - 8 * Math.sin(ang + 0.45));
        ctx.fill();
      }
    }
    // commit
    for (const c of s.cs) {
      const [x, y] = P(c);
      const live = reach.has(c.id), isNew = newIds.has(c.id);
      ctx.beginPath(); ctx.arc(x, y, R, 0, 7);
      ctx.fillStyle = isNew ? alpha(C.edge, 0.22) : live ? alpha(C.ink2, 0.18) : C.bg;
      ctx.fill();
      ctx.lineWidth = isNew ? 2.6 : 1.8;
      ctx.strokeStyle = isNew ? C.edge : live ? C.ink2 : C.grid2;
      ctx.setLineDash(live ? [] : [3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = `700 ${Math.round(R * 0.8)}px JetBrains Mono, monospace`;
      ctx.textAlign = "center";
      ctx.fillStyle = live ? C.ink : C.mid;
      ctx.fillText(c.id, x, y + 1);
      ctx.textAlign = "start";
    }
    // 이름표: 위 줄 commit은 위로, 아래 줄 commit은 아래로 쌓는다
    const stackAt = new Map<string, number>();
    const chip = (text: string, x: number, y: number, hot: boolean, dark: boolean) => {
      ctx.font = "700 11px JetBrains Mono, monospace";
      const w = ctx.measureText(text).width + 14, h = 19;
      const cx = Math.max(pad + w / 2, Math.min(gx1 - w / 2, x));
      rr(cx - w / 2, y - h / 2, w, h, 9.5);
      ctx.fillStyle = dark ? (hot ? C.edge : C.ink) : hot ? alpha(C.edge, 0.2) : C.bg;
      ctx.fill();
      ctx.lineWidth = hot ? 2 : 1.2;
      ctx.strokeStyle = hot ? C.edge : C.mid;
      if (!dark) ctx.stroke();
      ctx.fillStyle = dark ? C.bg : hot ? C.edge : C.ink;
      ctx.textAlign = "center";
      ctx.fillText(text, cx, y + 0.5);
      ctx.textAlign = "start";
    };
    const place = (id: string) => {
      const c = byId.get(id)!;
      const n = stackAt.get(id) ?? 0;
      stackAt.set(id, n + 1);
      const [x, y] = P(c);
      const dir = c.y === 0 ? -1 : 1;
      return [x, y + dir * (R + 14 + n * 23)] as [number, number];
    };
    for (const [name, id] of Object.entries(s.refs)) {
      const [x, y] = place(id);
      chip(name, x, y, movedRefs.has(name), false);
      if (name === s.head) { const [hx, hy] = place(id); chip("HEAD", hx, hy, headMoved, true); }
    }

    // reflog
    if (hasLog && s.reflog) {
      const lx = narrow ? pad : W * 0.6, ly = narrow ? H * 0.6 : 48, lw = narrow ? W - 2 * pad : W * 0.4 - pad;
      ctx.font = "600 12px Pretendard Variable, sans-serif";
      ctx.fillStyle = C.mid;
      ctx.fillText("git reflog main", lx, ly);
      ctx.font = "11px JetBrains Mono, monospace";
      const lh = Math.min(22, (H - ly - 20) / Math.max(s.reflog.length, 4));
      s.reflog.forEach((line, i) => {
        const y = ly + 22 + i * lh;
        if (s.mark === i) { ctx.fillStyle = alpha(C.edge, 0.18); rr(lx - 4, y - lh / 2 + 1, lw + 4, lh - 2, 6); ctx.fill(); }
        ctx.fillStyle = s.mark === i ? C.edge : i === 0 ? C.ink : C.mid;
        ctx.fillText(fit(line, lw), lx, y);
      });
    }
    ctx.textBaseline = "alphabetic";
  }
  function rr(x: number, y: number, w: number, h: number, r: number) {
    const c = f.ctx; c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
  }
  function fit(t: string, w: number) {
    if (f.ctx.measureText(t).width <= w) return t;
    let s = t; while (s.length > 4 && f.ctx.measureText(s + "…").width > w) s = s.slice(0, -1);
    return s + "…";
  }

  const pre = presets(Object.entries(SCENARIOS).map(([kk, v]) => ({ key: kk, label: v.label })), (kk) => { key = kk; S = states(key); k = 0; ps.input.max = String(S.length - 1); update(); }, key);
  pre.row.setAttribute("aria-label", "시나리오");
  const ps = param({ label: "단계", min: 0, max: S.length - 1, step: 1, value: 0, fmt: (v) => `${v}/${S.length - 1}`, aria: "명령 단계", onInput: (v) => { k = v; update(); } });
  const stepGrid = el("div", "param-grid");
  stepGrid.append(ps.row);
  const tools = el("div", "wframe-row");
  const mk = (t: string, aria: string, fn: () => void) => { const b = el("button", "preset"); b.type = "button"; b.textContent = t; b.setAttribute("aria-label", aria); b.addEventListener("click", fn); tools.append(b); return b; };
  mk("◀ 이전", "이전 명령", () => { k = Math.max(0, k - 1); update(); });
  mk("다음 ▶", "다음 명령 실행", () => { k = Math.min(S.length - 1, k + 1); update(); });
  const ro = readouts();
  f.panel.append(pre.row, stepGrid, tools, ro.row);

  function update() {
    ps.set(k);
    (ps.row.querySelector("output") as HTMLElement).textContent = `${k}/${S.length - 1}`;
    const s = S[k], prev = S[Math.max(0, k - 1)];
    const reach = reachable(s);
    const fresh = k > 0 ? s.cs.filter((c) => !prev.cs.some((q) => q.id === c.id)).map((c) => c.id) : [];
    const lost = s.cs.filter((c) => !reach.has(c.id)).map((c) => c.id);
    ro.set("cmd", k > 0 ? `$ ${s.cmd}` : "시작 상태", k > 0 ? "hot" : "");
    ro.set("new", fresh.length ? `새 commit ${fresh.join(", ")}` : "새 commit 없음");
    ro.set("lost", lost.length ? `이름표 없는 commit ${lost.join(", ")}` : "모든 commit에 이름표로 닿는다", lost.length ? "hot" : "ok");
    ro.set("head", `HEAD → ${s.head} (${s.refs[s.head]})`);
    draw();
  }
  f.onResize(draw);
  update();
}
