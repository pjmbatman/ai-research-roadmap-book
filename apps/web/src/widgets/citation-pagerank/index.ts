// 인용 그래프 PageRank (data/widgets.toml: citation-pagerank).
// 23.1장 "핵심 논문 찾기"의 가상의 작은 분야(논문 8편)를 그대로 쓴다. 노드 = 논문(가로 = 연도), 화살표 = 인용(새 논문 → 옛 논문).
// r = (1 − d)/n + d·P r, P의 열 i = 1/k_i(참고문헌이 없으면 1/n). 노드 넓이가 PageRank에 비례한다.
// 프리셋은 간선을 더하거나 뺀 변형이다. 노드를 누르면 그 논문의 참고문헌(backward)과 피인용(forward)이 보인다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

type Node = { id: string; year: number; y: number };
const BASE_NODES: Node[] = [
  { id: "원전15", year: 2015, y: 0.45 }, { id: "후속17", year: 2017, y: 0.1 }, { id: "후속18", year: 2018, y: 0.5 },
  { id: "반박19", year: 2019, y: 0.97 }, { id: "벤치20", year: 2020, y: 0.2 }, { id: "후속21", year: 2021, y: 0.68 },
  { id: "서베이23", year: 2023, y: 0.08 }, { id: "최신24", year: 2024, y: 0.45 },
];
export const BASE: Record<string, string[]> = {
  "후속17": ["원전15"], "후속18": ["원전15", "후속17"], "반박19": ["원전15", "후속18"],
  "벤치20": ["후속17", "후속18"], "후속21": ["후속18", "벤치20", "반박19"],
  "서베이23": ["후속17", "후속18", "반박19", "벤치20", "후속21"],
  "최신24": ["후속21", "벤치20", "원전15"],
};
const NEW: Node[] = [{ id: "새25a", year: 2025, y: 0.62 }, { id: "새25b", year: 2025, y: 0.8 }, { id: "새25c", year: 2025, y: 0.98 }];

type Pre = "base" | "drop" | "inflate" | "survey";
export function variant(k: Pre) {
  const cites: Record<string, string[]> = Object.fromEntries(Object.entries(BASE).map(([a, b]) => [a, [...b]]));
  let nodes = [...BASE_NODES];
  if (k === "drop") cites["후속18"] = ["원전15"];
  if (k === "survey") cites["서베이23"].push("원전15");
  if (k === "inflate") { nodes = [...nodes, ...NEW]; for (const nd of NEW) cites[nd.id] = ["반박19"]; }
  return { nodes, cites };
}

/** PageRank(power iteration)와 피인용 수, 수렴까지의 반복 수 */
export function pagerank(nodes: Node[], cites: Record<string, string[]>, d: number) {
  const n = nodes.length, idx = new Map(nodes.map((nd, i) => [nd.id, i]));
  const outs = nodes.map((nd) => (cites[nd.id] ?? []).map((t) => idx.get(t)!));
  const indeg = new Array(n).fill(0);
  outs.forEach((o) => o.forEach((j) => indeg[j]++));
  let r = new Array(n).fill(1 / n), iters = 0;
  for (; iters < 20000; iters++) {
    const nx = new Array(n).fill((1 - d) / n);
    outs.forEach((o, i) => {
      if (o.length === 0) for (let j = 0; j < n; j++) nx[j] += (d * r[i]) / n;
      else for (const j of o) nx[j] += (d * r[i]) / o.length;
    });
    const diff = nx.reduce((s, v, j) => s + Math.abs(v - r[j]), 0);
    r = nx;
    if (diff < 1e-12) { iters++; break; }
  }
  return { r, indeg, outs, iters };
}

const PRE_LABEL: Record<Pre, string> = { base: "본문", drop: "18→17 삭제", inflate: "새 3편→반박19", survey: "서베이→원전" };

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "인용 그래프 PageRank", "manipulate", narrow ? 0.95 : 0.6, "연도순으로 놓은 논문 노드와 인용 화살표, 노드 넓이가 PageRank에 비례한다");
  let pre: Pre = "base", d = 0.85, sel = "후속18";
  let g = variant(pre), res = pagerank(g.nodes, g.cites, d);

  const layout = () => {
    const { width: W, height: H } = f;
    const L = 26, R = 26, top = 34, bottom = H - 56;
    return g.nodes.map((nd) => ({ ...nd, x: L + ((nd.year - 2014.6) / 10.8) * (W - L - R), py: top + nd.y * (bottom - top) }));
  };
  const radius = (v: number) => (narrow ? 25 : 31) * Math.sqrt(v / 0.35);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const pos = layout(), si = g.nodes.findIndex((nd) => nd.id === sel);
    // 연도 축
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid; ctx.textBaseline = "alphabetic";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    const ay = H - 18;
    ctx.beginPath(); ctx.moveTo(14, ay - 10); ctx.lineTo(W - 14, ay - 10); ctx.stroke();
    for (let yr = 2015; yr <= 2025; yr += narrow ? 2 : 1) {
      const x = 26 + ((yr - 2014.6) / 10.8) * (W - 52), s = narrow ? `'${String(yr).slice(2)}` : String(yr);
      ctx.fillText(s, x - ctx.measureText(s).width / 2, ay + 4);
    }
    // 인용 화살표: 선택한 논문의 backward(참고문헌)는 진한 파랑, forward(피인용)는 코랄
    res.outs.forEach((o, i) => o.forEach((j) => {
      const a = pos[i], b = pos[j];
      const isBack = i === si, isFwd = j === si;
      const col = isFwd ? C.edge : isBack ? C.ink2 : C.grid2;
      const ang = Math.atan2(b.py - a.py, b.x - a.x);
      const ra = radius(res.r[i]) + 2, rb = radius(res.r[j]) + 3;
      const x0 = a.x + ra * Math.cos(ang), y0 = a.py + ra * Math.sin(ang), x1 = b.x - rb * Math.cos(ang), y1 = b.py - rb * Math.sin(ang);
      ctx.strokeStyle = col; ctx.lineWidth = isBack || isFwd ? 2.2 : 1.2;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.fillStyle = col; ctx.beginPath();
      ctx.moveTo(x1, y1); ctx.lineTo(x1 - 8 * Math.cos(ang - 0.4), y1 - 8 * Math.sin(ang - 0.4)); ctx.lineTo(x1 - 8 * Math.cos(ang + 0.4), y1 - 8 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
    }));
    // 노드: 넓이 ∝ PageRank. 이름은 위에, 값은 노드 안(작으면 아래)에. 글자에는 바탕색 테두리를 둘러 화살표 위에서도 읽히게 한다
    pos.forEach((p, i) => {
      ctx.fillStyle = alpha(C.ink2, i === si ? 0.85 : 0.5);
      ctx.beginPath(); ctx.arc(p.x, p.py, radius(res.r[i]), 0, 7); ctx.fill();
      if (i === si) { ctx.strokeStyle = C.edge; ctx.lineWidth = 3; ctx.stroke(); }
    });
    const halo = (text: string, x: number, y: number) => { ctx.strokeStyle = C.bg; ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.strokeText(text, x, y); ctx.fillText(text, x, y); };
    pos.forEach((p, i) => {
      const rad = radius(res.r[i]);
      ctx.fillStyle = C.ink; ctx.font = `${i === si ? 700 : 600} 12px Pretendard Variable, sans-serif`;
      const w = ctx.measureText(p.id).width;
      // 오른쪽 끝(2025년) 노드는 세로로 붙어 있어 이름을 왼쪽에 쓴다
      if (p.year >= 2025) halo(p.id, p.x - rad - 5 - w, p.py + 4);
      else halo(p.id, Math.max(2, Math.min(W - w - 2, p.x - w / 2)), p.py - rad - 5);
      ctx.font = "10px JetBrains Mono, monospace";
      const v = res.r[i].toFixed(3), vw = ctx.measureText(v).width;
      if (rad >= vw / 2 + 3) { ctx.fillStyle = C.ink; ctx.fillText(v, p.x - vw / 2, p.py + 3.5); }
      else { ctx.fillStyle = C.mid; halo(v, Math.max(2, Math.min(W - vw - 2, p.x - vw / 2)), p.py + rad + 11); }
    });
  }

  function update(rebuild = false) {
    if (rebuild) g = variant(pre);
    if (!g.nodes.some((nd) => nd.id === sel)) sel = "후속18";
    res = pagerank(g.nodes, g.cites, d);
    const order = g.nodes.map((_, i) => i).sort((a, b) => res.r[b] - res.r[a]);
    const top = order[0], si = g.nodes.findIndex((nd) => nd.id === sel);
    ro.set("top", `PageRank 1위 ${g.nodes[top].id} ${res.r[top].toFixed(3)}`, "hot");
    const maxIn = Math.max(...res.indeg);
    ro.set("cite", `피인용 최다 ${g.nodes.filter((_, i) => res.indeg[i] === maxIn).map((nd) => nd.id).join("·")} ${maxIn}회`);
    ro.set("sel", `${sel}: 피인용 ${res.indeg[si]} · PageRank ${res.r[si].toFixed(3)} (${order.indexOf(si) + 1}위)`, "ok");
    draw();
  }

  const pb = presets<Pre>((Object.keys(PRE_LABEL) as Pre[]).map((k) => ({ key: k, label: PRE_LABEL[k] })), (k) => { pre = k; update(true); }, pre);
  const pd = param({ label: "d", min: 0, max: 0.99, step: 0.01, value: d, aria: "damping factor d", onInput: (v) => { d = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pd.row);
  const ro = readouts();
  f.panel.append(pb.row, grid, ro.row);

  // 노드를 누르면 선택
  const hit = (e: PointerEvent | MouseEvent) => {
    const rc = f.canvas.getBoundingClientRect(), px = e.clientX - rc.left, py = e.clientY - rc.top;
    const pos = layout();
    let best = -1, bd = Infinity;
    pos.forEach((p, i) => { const dd = Math.hypot(p.x - px, p.py - py) - radius(res.r[i]); if (dd < bd) { bd = dd; best = i; } });
    return bd < 14 ? best : -1;
  };
  f.canvas.addEventListener("click", (e) => { const i = hit(e); if (i >= 0) { sel = g.nodes[i].id; update(); } });
  f.canvas.addEventListener("pointermove", (e) => { f.canvas.style.cursor = hit(e) >= 0 ? "pointer" : "default"; });
  f.canvas.tabIndex = 0;
  f.canvas.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const i = g.nodes.findIndex((nd) => nd.id === sel), n = g.nodes.length;
    sel = g.nodes[(i + (e.key === "ArrowRight" ? 1 : n - 1)) % n].id;
    update();
  });
  f.onResize(draw);
  update();
}
