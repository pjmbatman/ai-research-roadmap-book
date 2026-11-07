// temperature와 top-k·top-p (data/widgets.toml: sampling-temperature).
// 18.5장 "사용 방식": p_i(τ) = softmax(z/τ). top-k는 logit 상위 k개만 남기고(같은 값은 함께 남긴다, 본문 셀과 같다),
// top-p는 확률이 큰 순서로 누적 확률이 p에 처음 이를 때까지의 토큰만 남긴 뒤 다시 정규화한다. 적용 순서는 τ → top-k → top-p.
// 첫 프리셋은 본문 셀의 logit [2.0, 1.0, 0.5, −1.0]이고, 나머지 둘은 위젯에서 지어낸 가상 logit이다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

export function decode(logits: number[], T: number, k: number | null, p: number | null) {
  const z = logits.map((v) => v / T);
  const order = z.map((_, i) => i).sort((a, b) => z[b] - z[a]);
  const keep = z.map(() => true);
  if (k !== null && k < z.length) { const th = z[order[k - 1]]; z.forEach((v, i) => { if (v < th) keep[i] = false; }); }
  const soft = (mask: boolean[]) => {
    const m = Math.max(...z.filter((_, i) => mask[i]));
    const e = z.map((v, i) => (mask[i] ? Math.exp(v - m) : 0));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  };
  const full = soft(z.map(() => true));
  let q = soft(keep);
  if (p !== null && p < 1) {
    let acc = 0;
    const keep2 = z.map(() => false);
    for (const i of order) { if (!keep[i]) continue; keep2[i] = true; acc += q[i]; if (acc >= p - 1e-12) break; }
    q = soft(keep2);
  }
  const H = -q.reduce((s, v) => (v > 0 ? s + v * Math.log(v) : s), 0);
  return { full, q, H, n: q.filter((v) => v > 0).length, order };
}

const PRESETS = {
  book: { label: "본문 4개", z: [2.0, 1.0, 0.5, -1.0] },
  tail: { label: "긴 꼬리(가상)", z: [3.0, 1.6, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7] },
  tie: { label: "경합(가상)", z: [2.0, 1.9, 0.0, -0.5, -1.0, -1.5] },
} as const;
type PK = keyof typeof PRESETS;

export function mount(root: HTMLElement) {
  const f = frame(root, "temperature와 top-k·top-p", "manipulate", 0.46, "다음 토큰 후보별 샘플링 확률 막대: 자르기 전과 자른 뒤");
  let key: PK = "book";
  let z: number[] = [...PRESETS.book.z];
  let T = 1, k = z.length, p = 1;
  let res = decode(z, T, null, null);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 34, PR = 10, top = 22, bottom = H - 24;
    const V = z.length;
    const slot = (W - PL - PR) / V, bw = Math.min(slot * 0.7, 64);
    const Y = (v: number) => bottom - v * (bottom - top);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    for (const v of [0, 0.5, 1]) {
      ctx.beginPath(); ctx.moveTo(PL, Y(v)); ctx.lineTo(W - PR, Y(v)); ctx.stroke();
      ctx.fillStyle = C.mid; ctx.fillText(v.toFixed(1), 4, Y(v) + 4);
    }
    res.order.forEach((i, r) => {
      const x = PL + slot * (r + 0.5);
      const pf = res.full[i], pq = res.q[i];
      // 자르기 전 p_i(τ): 테두리만
      ctx.strokeStyle = alpha(C.mid, 0.8); ctx.lineWidth = 1; ctx.setLineDash([3, 2]);
      ctx.strokeRect(x - bw / 2, Y(pf), bw, bottom - Y(pf));
      ctx.setLineDash([]);
      // 자른 뒤 다시 정규화한 확률: 채움
      if (pq > 0) { ctx.fillStyle = r === 0 ? C.edge : alpha(C.edge, 0.7); ctx.fillRect(x - bw / 2, Y(pq), bw, bottom - Y(pq)); }
      else { ctx.strokeStyle = C.mid; ctx.beginPath(); ctx.moveTo(x - bw / 2, bottom - 3); ctx.lineTo(x + bw / 2, bottom - 3); ctx.stroke(); }
      ctx.fillStyle = pq > 0 ? C.ink : C.mid;
      const lab = `${r + 1}`, lw = ctx.measureText(lab).width;
      ctx.fillText(lab, x - lw / 2, bottom + 14);
      if (bw >= 30 && pq > 0) { const t = pq.toFixed(3), tw = ctx.measureText(t).width; ctx.fillText(t, x - tw / 2, Math.max(top - 6, Y(Math.max(pq, pf)) - 5)); }
    });
    ctx.fillStyle = C.mid;
    ctx.fillText("순위", W - PR - ctx.measureText("순위").width, top - 8);
  }

  let raf = 0;
  function update() {
    res = decode(z, T, k < z.length ? k : null, p < 1 ? p : null);
    const top1 = res.q[res.order[0]];
    const tail = res.order.slice(3).reduce((s, i) => s + res.q[i], 0);
    ro.set("H", `H = ${res.H.toFixed(3)} nat`, "hot");
    ro.set("eff", `유효 후보 수 e^H ${Math.exp(res.H).toFixed(2)}`);
    ro.set("n", `남은 후보 ${res.n} / ${z.length}`);
    ro.set("top", `1위 ${top1.toFixed(3)}`);
    ro.set("tail", `4위 이하 합 ${tail.toFixed(3)}`);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const wide = <P extends { row: HTMLElement }>(q: P) => { q.row.style.gridTemplateColumns = "3.4em minmax(0, 1fr) 3.4em"; return q; };
  const pT = wide(param({ label: "τ", min: 0.05, max: 3, step: 0.05, value: T, fmt: (v) => v.toFixed(2), aria: "temperature τ", onInput: (v) => { T = v; update(); } }));
  const pk = wide(param({ label: "top-k", min: 1, max: 4, step: 1, value: k, fmt: (v) => (v >= z.length ? "끔" : `${v}`), aria: "top-k (최댓값이면 끔)", onInput: (v) => { k = v; update(); } }));
  const pp = wide(param({ label: "top-p", min: 0.05, max: 1, step: 0.05, value: p, fmt: (v) => (v >= 1 ? "끔" : v.toFixed(2)), aria: "top-p (1이면 끔)", onInput: (v) => { p = v; update(); } }));
  const pre = presets((Object.keys(PRESETS) as PK[]).map((q) => ({ key: q, label: PRESETS[q].label })), (q) => {
    key = q; z = [...PRESETS[q].z]; k = z.length;
    pk.input.max = String(z.length); pk.set(k);
    update();
  }, key);
  const grid = el("div", "param-grid");
  grid.append(pT.row, pk.row, pp.row);
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(140px, 1fr))";   // 수업 시트 폭에서도 두 칸
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
