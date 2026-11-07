// 계산 그래프와 역전파 (data/widgets.toml: autodiff-graph).
// 작은 식을 계산 그래프로 펼쳐 forward(값)와 backward(adjoint v̄ = ∂f/∂v)를 노드 하나씩 재생한다.
// backward의 한 단계는 그 노드를 쓰는 노드들의 adjoint × 국소 미분을 더한다(여러 곳에 쓰인 입력은 기울기가 합쳐진다).
// 끝나면 같은 기울기를 중심 차분으로 구해 상대오차를 보인다.
import { frame, param, presets, readouts, el, C, reducedMotion } from "../frame";
import { pageState } from "../../client/store";

interface GNode {
  id: string; name: string; at: [number, number];
  op?: string; inputs?: string[]; expr?: string;
  f?: (a: number[]) => number; d?: (a: number[]) => number[]; dsym?: string[];
}
interface Graph { label: string; nodes: GNode[]; params: { id: string; min: number; max: number; step: number; value: number }[]; consts: Record<string, number> }

export const GRAPHS: Record<string, Graph> = {
  basic: {
    label: "ln x₁+x₁x₂−sin x₂",
    nodes: [
      { id: "x1", name: "x₁", at: [0, 0.2] },
      { id: "x2", name: "x₂", at: [0, 0.8] },
      { id: "v1", name: "v₁", at: [1, 0], op: "ln", inputs: ["x1"], expr: "ln x₁", f: ([a]) => Math.log(a), d: ([a]) => [1 / a], dsym: ["(1/x₁)"] },
      { id: "v2", name: "v₂", at: [1, 0.5], op: "×", inputs: ["x1", "x2"], expr: "x₁·x₂", f: ([a, b]) => a * b, d: ([a, b]) => [b, a], dsym: ["x₂", "x₁"] },
      { id: "v3", name: "v₃", at: [1, 1], op: "sin", inputs: ["x2"], expr: "sin x₂", f: ([a]) => Math.sin(a), d: ([a]) => [Math.cos(a)], dsym: ["cos x₂"] },
      { id: "v4", name: "v₄", at: [2, 0.25], op: "+", inputs: ["v1", "v2"], expr: "v₁ + v₂", f: ([a, b]) => a + b, d: () => [1, 1], dsym: ["1", "1"] },
      { id: "f", name: "f", at: [3, 0.6], op: "−", inputs: ["v4", "v3"], expr: "v₄ − v₃", f: ([a, b]) => a - b, d: () => [1, -1], dsym: ["1", "(−1)"] },
    ],
    params: [{ id: "x1", min: 0.2, max: 4, step: 0.1, value: 2 }, { id: "x2", min: -4, max: 6, step: 0.1, value: 5 }],
    consts: {},
  },
  neuron: {
    label: "(tanh(wx+b)−y)²",
    nodes: [
      { id: "w", name: "w", at: [0, 0] },
      { id: "x", name: "x", at: [0, 0.5] },
      { id: "b", name: "b", at: [1, 0.85] },
      { id: "u", name: "u", at: [1, 0.25], op: "×", inputs: ["w", "x"], expr: "w·x", f: ([a, b]) => a * b, d: ([a, b]) => [b, a], dsym: ["x", "w"] },
      { id: "z", name: "z", at: [2, 0.5], op: "+", inputs: ["u", "b"], expr: "u + b", f: ([a, b]) => a + b, d: () => [1, 1], dsym: ["1", "1"] },
      { id: "a", name: "a", at: [3, 0.3], op: "tanh", inputs: ["z"], expr: "tanh z", f: ([a]) => Math.tanh(a), d: ([a]) => [1 - Math.tanh(a) ** 2], dsym: ["(1 − a²)"] },
      { id: "y", name: "y", at: [3, 0.95] },
      { id: "L", name: "L", at: [4, 0.6], op: "( )²", inputs: ["a", "y"], expr: "(a − y)²", f: ([a, y]) => (a - y) ** 2, d: ([a, y]) => [2 * (a - y), -2 * (a - y)], dsym: ["2(a − y)", "(−2)(a − y)"] },
    ],
    params: [{ id: "w", min: -2, max: 2, step: 0.05, value: 0.5 }, { id: "b", min: -2, max: 2, step: 0.05, value: 0.1 }],
    consts: { x: 1.5, y: 0.8 },
  },
};

type Step = { kind: "input" } | { kind: "fwd"; id: string } | { kind: "seed" } | { kind: "bwd"; id: string };
export function steps(g: Graph): Step[] {
  const ops = g.nodes.filter((n) => n.op);
  const out = ops[ops.length - 1].id;
  return [{ kind: "input" }, ...ops.map((n) => ({ kind: "fwd" as const, id: n.id })), { kind: "seed" },
    ...[...g.nodes].reverse().filter((n) => n.id !== out).map((n) => ({ kind: "bwd" as const, id: n.id }))];
}
/** 입력값으로 그래프 전체를 forward·backward 한다(노드는 위상 순서로 적혀 있다). */
export function run(g: Graph, inputs: Record<string, number>) {
  const val: Record<string, number> = {}, adj: Record<string, number> = {};
  for (const n of g.nodes) val[n.id] = n.op ? n.f!(n.inputs!.map((i) => val[i])) : inputs[n.id];
  const out = g.nodes.filter((n) => n.op).pop()!.id;
  g.nodes.forEach((n) => (adj[n.id] = 0));
  adj[out] = 1;
  for (const n of [...g.nodes].reverse()) {
    if (!n.op) continue;
    const loc = n.d!(n.inputs!.map((i) => val[i]));
    n.inputs!.forEach((i, k) => (adj[i] += adj[n.id] * loc[k]));
  }
  return { val, adj, out };
}

const num = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(3)).replace("-", "−");

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "계산 그래프와 역전파", "stepper", root.clientWidth < 480 ? 0.95 : 0.6, "계산 그래프의 노드별 forward 값과 backward 기울기를 단계별로 그린 그림");
  let key = attrs.preset in GRAPHS ? attrs.preset : "basic";
  let g = GRAPHS[key];
  let inputs: Record<string, number> = {};
  let S = steps(g);
  let k = 0;
  let timer = 0;

  const byId = (id: string) => g.nodes.find((n) => n.id === id)!;
  const consumers = (id: string) => g.nodes.filter((n) => n.inputs?.includes(id));

  function state() {
    const full = run(g, inputs);
    const known = new Set<string>(), back = new Set<string>();
    for (let i = 0; i <= k; i++) {
      const s = S[i];
      if (s.kind === "input") g.nodes.forEach((n) => { if (!n.op) known.add(n.id); });
      else if (s.kind === "fwd") known.add(s.id);
      else if (s.kind === "seed") back.add(full.out);
      else back.add(s.id);
    }
    return { ...full, known, back };
  }

  function layout() {
    const cols = Math.max(...g.nodes.map((n) => n.at[0])) + 1;
    const r = Math.max(14, Math.min(22, f.width / (cols * 5.2)));
    const fs = Math.max(9.5, Math.min(12, f.width / Math.max(1, cols - 1) / 7.5));
    const padX = Math.max(r + 6, fs * 3.6);
    const top = r + 22, bottom = r + 2 * fs + 14;
    const P = (n: GNode): [number, number] => [padX + (n.at[0] / Math.max(1, cols - 1)) * (f.width - 2 * padX), top + n.at[1] * (f.height - top - bottom)];
    return { r, P, fs };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const st = state();
    const { r, P, fs } = layout();
    const cur = S[k];
    const curId = cur.kind === "fwd" || cur.kind === "bwd" ? cur.id : cur.kind === "seed" ? st.out : null;
    // 엣지
    for (const n of g.nodes) {
      if (!n.inputs) continue;
      const loc = n.d!(n.inputs.map((i) => st.val[i]));
      n.inputs.forEach((i, j) => {
        const a = P(byId(i)), b = P(n);
        const fwdHot = cur.kind === "fwd" && cur.id === n.id;
        const bwdHot = cur.kind === "bwd" && cur.id === i;
        const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
        const s0: [number, number] = [a[0] + r * Math.cos(ang), a[1] + r * Math.sin(ang)];
        const s1: [number, number] = [b[0] - r * Math.cos(ang), b[1] - r * Math.sin(ang)];
        ctx.strokeStyle = fwdHot || bwdHot ? C.edge : st.known.has(n.id) ? C.ink2 : C.grid2;
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = fwdHot || bwdHot ? 2.4 : 1.4;
        ctx.beginPath(); ctx.moveTo(...s0); ctx.lineTo(...s1); ctx.stroke();
        // forward는 입력 → 노드, backward는 노드 → 입력 방향 화살표
        head(bwdHot ? s0 : s1, bwdHot ? ang + Math.PI : ang, bwdHot ? 9 : 7);
        if (bwdHot) {
          const m: [number, number] = [(s0[0] + s1[0]) / 2, (s0[1] + s1[1]) / 2];
          const t = `×${num(loc[j])}`;
          ctx.font = `600 ${fs}px JetBrains Mono, monospace`;
          const w = ctx.measureText(t).width;
          ctx.fillStyle = C.bg; ctx.fillRect(m[0] - w / 2 - 3, m[1] - fs * 0.8, w + 6, fs * 1.4);
          ctx.fillStyle = C.edge; ctx.fillText(t, m[0] - w / 2, m[1] + fs * 0.35);
        }
      });
    }
    // 노드
    ctx.textAlign = "center";
    for (const n of g.nodes) {
      const p = P(n);
      const hot = n.id === curId || (cur.kind === "input" && !n.op);
      const known = st.known.has(n.id);
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7);
      ctx.fillStyle = known ? C.fill : C.bg; ctx.fill();
      ctx.lineWidth = hot ? 3 : 1.5;
      ctx.strokeStyle = hot ? C.edge : known ? C.ink2 : C.grid2; ctx.stroke();
      ctx.fillStyle = known ? C.ink : C.mid;
      const inside = n.op ?? n.name;
      ctx.font = `700 ${inside.length > 2 ? Math.max(9, r * 0.55) : r * 0.8}px JetBrains Mono, monospace`;
      ctx.fillText(inside, p[0], p[1] + r * 0.27);
      ctx.font = `${fs}px JetBrains Mono, monospace`;
      if (n.op) { ctx.fillStyle = C.mid; ctx.fillText(n.name, p[0], p[1] - r - 6); }
      if (known) { ctx.fillStyle = C.ink; ctx.fillText(num(st.val[n.id]), p[0], p[1] + r + fs + 3); }
      if (st.back.has(n.id)) {
        // adjoint v̄: 이름 위에 막대를 직접 긋는다(결합 문자는 첨자와 겹친다)
        ctx.fillStyle = C.edge; ctx.strokeStyle = C.edge; ctx.font = `700 ${fs}px JetBrains Mono, monospace`;
        const t = `${n.name} ${num(st.adj[n.id])}`, y = p[1] + r + 2 * fs + 8;
        ctx.fillText(t, p[0], y);
        const x0 = p[0] - ctx.measureText(t).width / 2, cw = ctx.measureText(n.name[0]).width;
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(x0, y - fs * 0.88); ctx.lineTo(x0 + cw, y - fs * 0.88); ctx.stroke();
      }
    }
    ctx.textAlign = "start";
    ctx.lineWidth = 1;
  }
  function head(p: [number, number], ang: number, s: number) {
    const { ctx } = f;
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    ctx.lineTo(p[0] - s * Math.cos(ang - 0.42), p[1] - s * Math.sin(ang - 0.42));
    ctx.lineTo(p[0] - s * Math.cos(ang + 0.42), p[1] - s * Math.sin(ang + 0.42));
    ctx.fill();
  }

  function describe(st: ReturnType<typeof state>): [string, string] {
    const s = S[k];
    if (s.kind === "input") return ["입력", g.nodes.filter((n) => !n.op).map((n) => `${n.name} = ${num(st.val[n.id])}`).join(", ")];
    if (s.kind === "fwd") {
      const n = byId(s.id);
      return [`${n.name} = ${n.expr}`, `= ${num(st.val[n.id])}`];
    }
    const o = byId(st.out).name, D = (name: string) => `∂${o}/∂${name}`;
    if (s.kind === "seed") return [`${D(o)}`, "= 1"];
    const n = byId(s.id), cs = consumers(n.id);
    const sym = cs.map((c) => `${D(c.name)}·${c.dsym![c.inputs!.indexOf(n.id)]}`).join(" + ");
    const nums = cs.map((c) => `${num(st.adj[c.id])}×${num(c.d!(c.inputs!.map((i) => st.val[i]))[c.inputs!.indexOf(n.id)])}`).join(" + ");
    return [`${D(n.name)} = ${sym}`, `= ${nums} = ${num(st.adj[n.id])}`];
  }

  // 조작
  const pre = presets(Object.entries(GRAPHS).map(([key, gg]) => ({ key, label: gg.label })), (kk) => { key = kk; setGraph(); }, key);
  const ps = param({ label: "단계", min: 0, max: S.length - 1, step: 1, value: k, fmt: (v) => `${v}/${S.length - 1}`, aria: "재생 단계", onInput: (v) => { stop(); k = v; update(); } });
  const stepGrid = el("div", "param-grid");
  stepGrid.append(ps.row);
  const tools = el("div", "wframe-row");
  const mk = (t: string, aria: string, fn: () => void) => { const b = el("button", "preset"); b.type = "button"; b.textContent = t; b.setAttribute("aria-label", aria); b.addEventListener("click", fn); tools.append(b); return b; };
  mk("◀ 이전", "이전 단계", () => { stop(); k = Math.max(0, k - 1); update(); });
  mk("다음 ▶", "다음 단계", () => { stop(); k = Math.min(S.length - 1, k + 1); update(); });
  const playBtn = mk("▶ 재생", "처음부터 끝까지 재생", () => play());
  const inGrid = el("div", "param-grid");
  const ro = readouts();
  f.panel.append(pre.row, stepGrid, tools, inGrid, ro.row);

  function setGraph() {
    stop();
    g = GRAPHS[key];
    S = steps(g);
    k = 0;
    inputs = { ...g.consts };
    inGrid.innerHTML = "";
    g.params.forEach((p) => {
      inputs[p.id] = p.value;
      const node = byId(p.id);
      const q = param({ label: node.name, min: p.min, max: p.max, step: p.step, value: p.value, aria: `입력 ${node.name}`, onInput: (v) => { inputs[p.id] = v; update(); } });
      inGrid.append(q.row);
    });
    ps.input.max = String(S.length - 1);
    update();
  }

  function update() {
    ps.set(k);
    const st = state();
    const [a, b] = describe(st);
    const s = S[k];
    ro.set("phase", s.kind === "fwd" || s.kind === "input" ? "forward" : "backward", s.kind === "fwd" || s.kind === "input" ? "" : "hot");
    ro.set("sym", a, "hot");
    ro.set("num", b);
    const outN = byId(st.out);
    ro.set("f", st.known.has(st.out) ? `${outN.name} = ${num(st.val[st.out])}` : `${outN.name} = ?`);
    const done = k === S.length - 1;
    const grads = g.params.map((p) => `∂${outN.name}/∂${byId(p.id).name} = ${st.back.has(p.id) ? num(st.adj[p.id]) : "?"}`).join(" · ");
    ro.set("grad", grads, done ? "ok" : "");
    if (done) {
      // 중심 차분과 비교
      let worst = 0;
      for (const p of g.params) {
        const h = 1e-6 * Math.max(1, Math.abs(inputs[p.id]));
        const fp = run(g, { ...inputs, [p.id]: inputs[p.id] + h }).val[st.out];
        const fm = run(g, { ...inputs, [p.id]: inputs[p.id] - h }).val[st.out];
        const fd = (fp - fm) / (2 * h), an = st.adj[p.id];
        worst = Math.max(worst, Math.abs(an - fd) / Math.max(Math.abs(an), Math.abs(fd), 1e-12));
      }
      ro.set("check", `중심 차분과 상대오차 ${worst.toExponential(1)}`, worst < 1e-5 ? "ok" : "hot");
    } else ro.set("check", "중심 차분 비교: 끝 단계에서");
    pageState.set("autodiff", { graph: key, inputs: { ...inputs }, grad: Object.fromEntries(g.params.map((p) => [p.id, st.adj[p.id]])) }, root);
    draw();
  }

  function stop() { clearTimeout(timer); timer = 0; playBtn.textContent = "▶ 재생"; }
  function play() {
    if (timer) { stop(); return; }
    if (reducedMotion()) { k = S.length - 1; update(); return; }
    if (k === S.length - 1) k = 0;
    playBtn.textContent = "❚❚ 멈춤";
    const tick = () => {
      if (!f.canvas.isConnected) return;
      update();
      if (k >= S.length - 1) { stop(); return; }
      k++;
      timer = window.setTimeout(tick, 800);
    };
    timer = window.setTimeout(tick, 0);
  }

  f.onResize(draw);
  setGraph();
}
