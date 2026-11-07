// 다목적 Pareto front (data/widgets.toml: pareto-front). 25.4장 다목적과 제약 절.
// 가상의 후보 조건 60개가 두 목적(수율, 순도, 둘 다 클수록 좋다)을 가진다. front 가운데가 오목하게 들어가 있다.
// 가중합 해 = argmax w·수율 + (1 − w)·순도, 단 순도 ≥ c인 후보 중에서. c = 0이면 순수한 가중합이다.
// 가중합만으로 닿는 front 점은 w를 0…1로 훑어 argmax가 되는 점들이다(볼록 껍질 위의 점). 오목한 부분의 점은 어떤 w로도 닿지 않고,
// 순도 하한 제약(ε-제약)을 걸면 닿는다.
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";

export type P = { a: number; b: number };   // a = 수율, b = 순도

const bound = (x: number) => 0.95 * Math.sqrt(Math.max(0, 1 - x * x)) - 0.12 * Math.exp(-(((x - 0.55) / 0.16) ** 2));
export function candidates(seed = 5): P[] {
  const r = rng(seed);
  const pts: P[] = [];
  for (let i = 0; i < 14; i++) { const a = 0.06 + (0.88 * i) / 13 + 0.02 * (r.u() - 0.5); pts.push({ a, b: bound(a) - 0.012 * r.u() }); }
  for (let i = 0; i < 46; i++) { const a = 0.04 + 0.9 * r.u(); pts.push({ a, b: Math.max(0.03, bound(a) * (0.35 + 0.58 * r.u())) }); }
  return pts;
}
/** 지배되지 않는 점: 두 목적 모두 같거나 크고 하나 이상에서 큰 다른 점이 없다 */
export function front(pts: P[]) {
  return pts.map((p, i) => !pts.some((q, j) => j !== i && q.a >= p.a && q.b >= p.b && (q.a > p.a || q.b > p.b)));
}
/** 순도 ≥ c 안에서 가중합 최대 */
export function argmaxW(pts: P[], w: number, c = 0) {
  let best = -1, s = -Infinity;
  pts.forEach((p, i) => { if (p.b < c) return; const v = w * p.a + (1 - w) * p.b; if (v > s + 1e-12) { s = v; best = i; } });
  return best;
}
/** 제약 없이 어떤 w로든 argmax가 되는 점들 */
export function supported(pts: P[]) {
  const hit = new Set<number>();
  for (let k = 0; k <= 2000; k++) hit.add(argmaxW(pts, k / 2000));
  return hit;
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "다목적 Pareto front", "manipulate", narrow ? 0.95 : 0.56, "가상 후보 조건의 수율과 순도, 지배되지 않는 front, 가중합 해와 순도 제약");
  const pts = candidates();
  const isF = front(pts);
  const sup = supported(pts);
  const fIdx = pts.map((_, i) => i).filter((i) => isF[i]).sort((i, j) => pts[i].a - pts[j].a);
  const unsup = fIdx.filter((i) => !sup.has(i));
  const dentC = unsup.length ? pts[unsup[Math.floor(unsup.length / 2)]].b : 0.5;
  let w = 0.5, c = 0;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const left = 42, bottom = H - 30, top = 10;
    const right = W - 12;
    const X = (a: number) => left + a * (right - left);
    const Y = (b: number) => bottom - b * (bottom - top);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.lineWidth = 1;
    ctx.strokeStyle = C.faint; ctx.strokeRect(left, top, right - left, bottom - top);
    // 제약: 순도 < c 영역
    if (c > 0) {
      ctx.fillStyle = alpha(C.mid, 0.14); ctx.fillRect(left, Y(c), right - left, bottom - Y(c));
      ctx.strokeStyle = C.mid; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(left, Y(c)); ctx.lineTo(right, Y(c)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.textAlign = "end"; ctx.fillText(`순도 ≥ ${c.toFixed(2)}`, right - 6, Y(c) - 5); ctx.textAlign = "start";
    }
    // front 계단선(지배 영역의 경계)
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.6;
    ctx.beginPath();
    fIdx.forEach((i, k) => {
      const p = pts[i];
      if (!k) { ctx.moveTo(X(0), Y(p.b)); ctx.lineTo(X(p.a), Y(p.b)); }
      else { ctx.lineTo(X(pts[fIdx[k - 1]].a), Y(p.b)); ctx.lineTo(X(p.a), Y(p.b)); }
    });
    ctx.lineTo(X(pts[fIdx[fIdx.length - 1]].a), Y(0));
    ctx.globalAlpha = 0.6; ctx.stroke(); ctx.globalAlpha = 1;
    // 가중합 해와 등고선 w·a + (1 − w)·b = const
    const bi = argmaxW(pts, w, c);
    if (bi >= 0) {
      const p = pts[bi], v = w * p.a + (1 - w) * p.b;
      ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();
      ctx.strokeStyle = C.edge; ctx.lineWidth = 1.4; ctx.setLineDash([6, 4]);
      ctx.beginPath();
      if (w < 1 - 1e-9) { ctx.moveTo(X(-0.1), Y((v + 0.1 * w) / (1 - w))); ctx.lineTo(X(1.1), Y((v - 1.1 * w) / (1 - w))); }
      else { ctx.moveTo(X(p.a), top); ctx.lineTo(X(p.a), bottom); }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.restore();
    }
    // 점
    pts.forEach((p, i) => {
      if (isF[i]) return;
      ctx.fillStyle = alpha(C.mid, p.b < c ? 0.3 : 0.75);
      ctx.beginPath(); ctx.arc(X(p.a), Y(p.b), 3, 0, 7); ctx.fill();
    });
    fIdx.forEach((i) => {
      const p = pts[i];
      ctx.globalAlpha = p.b < c ? 0.35 : 1;
      if (sup.has(i)) { ctx.fillStyle = C.ink2; ctx.beginPath(); ctx.arc(X(p.a), Y(p.b), 4.5, 0, 7); ctx.fill(); }
      else { ctx.fillStyle = C.bg; ctx.strokeStyle = C.ink2; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(p.a), Y(p.b), 4.5, 0, 7); ctx.fill(); ctx.stroke(); }
      ctx.globalAlpha = 1;
    });
    if (bi >= 0) {
      const p = pts[bi];
      ctx.strokeStyle = C.edge; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(X(p.a), Y(p.b), 8, 0, 7); ctx.stroke();
    }
    // 축
    ctx.fillStyle = C.mid; ctx.textAlign = "center";
    for (const v of [0, 0.5, 1]) ctx.fillText(String(v), X(v), bottom + 13);
    ctx.fillText("수율 (가상)", X(0.75), bottom + 13);
    ctx.textAlign = "right";
    for (const v of [0, 0.5, 1]) ctx.fillText(String(v), left - 5, Y(v) + 4);
    ctx.save(); ctx.translate(left - 28, Y(0.5)); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("순도 (가상)", 0, 0); ctx.restore();
    ctx.textAlign = "start";
    // 범례(오른쪽 위)
    const items: [string, (x: number, y: number) => void][] = [
      ["가중합으로 닿는 front", (x, y) => { ctx.fillStyle = C.ink2; ctx.beginPath(); ctx.arc(x + 5, y, 4, 0, 7); ctx.fill(); }],
      ["닿지 않는 front", (x, y) => { ctx.fillStyle = C.bg; ctx.strokeStyle = C.ink2; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x + 5, y, 4, 0, 7); ctx.fill(); ctx.stroke(); }],
      ["가중합 해", (x, y) => { ctx.strokeStyle = C.edge; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x + 5, y, 5, 0, 7); ctx.stroke(); }],
    ];
    const lw = Math.max(...items.map(([t]) => ctx.measureText(t).width)) + 26;
    const lx = left + 8, ly = bottom - items.length * 17 - 8;
    ctx.fillStyle = alpha(C.bg, 0.85); ctx.fillRect(lx - 4, ly, lw + 6, items.length * 17 + 6);
    items.forEach(([t, mark], k) => { const y = ly + 14 + k * 17; mark(lx, y - 4); ctx.fillStyle = C.ink; ctx.fillText(t, lx + 16, y); });
  }

  function update() {
    const bi = argmaxW(pts, w, c);
    ro.set("front", `front ${fIdx.length}개 · 가중합 도달 ${fIdx.filter((i) => sup.has(i)).length}개`);
    if (bi < 0) ro.set("sol", "제약을 만족하는 후보 없음", "hot");
    else {
      const p = pts[bi];
      ro.set("sol", `해: 수율 ${p.a.toFixed(3)} · 순도 ${p.b.toFixed(3)}`, sup.has(bi) ? "" : "ok");
    }
    ro.set("kind", bi >= 0 && !sup.has(bi) ? "오목부 점" : bi >= 0 && isF[bi] ? "볼록부 점" : "front 밖", bi >= 0 && !sup.has(bi) ? "ok" : "");
    draw();
  }
  const pw = param({ label: "w", min: 0, max: 1, step: 0.01, value: w, fmt: (v) => v.toFixed(2), aria: "가중합에서 수율의 가중치 w(순도의 가중치는 1 − w)", onInput: (v) => { w = v; pre.select(null); update(); } });
  const pc = param({ label: "c", min: 0, max: 0.95, step: 0.01, value: c, fmt: (v) => v.toFixed(2), aria: "순도 제약의 하한 c", onInput: (v) => { c = v; pre.select(null); update(); } });
  type Key = "sum" | "eps";
  const pre = presets<Key>([{ key: "sum", label: "가중합만 (w = 0.5)" }, { key: "eps", label: "수율 최대 + 순도 하한" }], (k) => {
    if (k === "sum") { w = 0.5; c = 0; } else { w = 1; c = Math.floor(dentC * 100) / 100; }
    pw.set(w); pc.set(c); update();
  }, "sum");
  const grid = el("div", "param-grid");
  grid.append(pw.row, pc.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
