// 브로드캐스팅 규칙 (data/widgets.toml: broadcasting-shapes).
// 5.3장의 세 규칙을 칸 그림으로 본다: (1) shape을 뒤에서부터 맞추고 축이 적은 쪽 앞에 길이 1을 붙인다,
// (2) 각 축의 두 길이는 같거나 한쪽이 1이어야 한다(아니면 ValueError), (3) 길이 1인 축은 상대 길이만큼 늘어난 것처럼 읽는다(stride 0).
// 프리셋은 본문의 예(행 평균 빼기, 바깥 곱, 열벡터 − 1차원 배열의 함정, 쌍 거리)와 오류가 나는 경우다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

type Sym = "N" | "D" | "M";
type Dim = Sym | 1;
const PRESETS: Record<string, { label: string; a: Dim[]; b: Dim[]; op: string }> = {
  center: { label: "(N, D) − (D,)", a: ["N", "D"], b: ["D"], op: "−" },
  outer: { label: "(N, 1) * (1, M)", a: ["N", 1], b: [1, "M"], op: "*" },
  trap: { label: "(N, 1) − (N,)", a: ["N", 1], b: ["N"], op: "−" },
  pair: { label: "(N, 1, D) − (1, N, D)", a: ["N", 1, "D"], b: [1, "N", "D"], op: "−" },
  error: { label: "(N, D) − (N,)", a: ["N", "D"], b: ["N"], op: "−" },
};

export interface Axis { a: number | null; b: number | null; ap: number; bp: number; out: number | null; stretchA: boolean; stretchB: boolean; bad: boolean }
/** NumPy 규칙대로 축마다 판정한다. 반환: 축 목록(앞에서 뒤로), 결과 shape(오류면 null), 오류 축 번호(음수, 뒤에서부터) */
export function broadcast(a: number[], b: number[]) {
  const nd = Math.max(a.length, b.length);
  const axes: Axis[] = [];
  let ok = true, badAxis = 0;
  for (let i = 0; i < nd; i++) {
    const ia = i - (nd - a.length), ib = i - (nd - b.length);
    const av = ia >= 0 ? a[ia] : null, bv = ib >= 0 ? b[ib] : null;
    const ap = av ?? 1, bp = bv ?? 1;
    const bad = ap !== bp && ap !== 1 && bp !== 1;
    if (bad) { ok = false; if (!badAxis) badAxis = i - nd; }
    const out = bad ? null : Math.max(ap, bp);
    axes.push({ a: av, b: bv, ap, bp, out, stretchA: !bad && ap === 1 && bp > 1, stretchB: !bad && bp === 1 && ap > 1, bad });
  }
  return { axes, shape: ok ? axes.map((x) => x.out!) : null, badAxis };
}
const fmtShape = (s: number[]) => `(${s.join(", ")}${s.length === 1 ? "," : ""})`;

export function mount(root: HTMLElement) {
  const f = frame(root, "브로드캐스팅 규칙", "manipulate", ((w) => (w < 400 ? 1.05 : w < 560 ? 0.8 : 0.56))(root.clientWidth || 600), "두 배열의 shape을 오른쪽으로 맞춘 표와, 길이 1인 축이 늘어나는 모습을 칸으로 그린 그림");
  let key = "trap";
  const val: Record<Sym, number> = { N: 5, D: 3, M: 4 };
  const dims = (d: Dim[]) => d.map((x) => (x === 1 ? 1 : val[x]));

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const pr = PRESETS[key];
    const A = dims(pr.a), Bs = dims(pr.b), r = broadcast(A, Bs);
    const nd = r.axes.length;
    const pad = 14;
    // 1) shape 정렬표
    const labW = 44, colW = Math.min(78, (W - 2 * pad - labW) / nd), cellH = 26;
    const tx0 = W - pad - colW * nd, ty0 = 26;
    ctx.textBaseline = "middle";
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.mid;
    ctx.fillText("뒤에서부터 맞춘 shape", pad, 12);
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.textAlign = "center";
    r.axes.forEach((_, i) => ctx.fillText(`축 ${i - nd}`, tx0 + (i + 0.5) * colW, ty0 + 4));
    const rowsY = [ty0 + 24, ty0 + 24 + cellH + 6, ty0 + 24 + 2 * (cellH + 6) + 8];
    ctx.textAlign = "start";
    ctx.font = "700 12px JetBrains Mono, monospace";
    ["a", "b", "결과"].forEach((t, j) => { ctx.fillStyle = C.ink; ctx.fillText(t, tx0 - labW, rowsY[j] + cellH / 2); });
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(tx0 - labW, rowsY[2] - 7); ctx.lineTo(W - pad, rowsY[2] - 7); ctx.stroke();
    r.axes.forEach((ax, i) => {
      const x = tx0 + i * colW + 3, w = colW - 6;
      const cell = (y: number, v: number | null, padded: boolean, stretch: boolean, bad: boolean, outLen: number) => {
        box(x, y, w, cellH, 7);
        ctx.fillStyle = bad ? alpha(C.edge, 0.3) : stretch ? alpha(C.edge, 0.16) : padded ? C.bg : alpha(C.ink2, 0.16);
        ctx.fill();
        ctx.setLineDash(padded ? [3, 3] : []);
        ctx.strokeStyle = bad || stretch ? C.edge : padded ? C.mid : C.ink2;
        ctx.lineWidth = bad ? 2 : 1.2;
        ctx.stroke(); ctx.setLineDash([]);
        ctx.font = `${bad || stretch ? 700 : 600} 12px JetBrains Mono, monospace`;
        ctx.fillStyle = bad || stretch ? C.edge : padded ? C.mid : C.ink;
        ctx.textAlign = "center";
        ctx.fillText(stretch ? `1→${outLen}` : String(v ?? 1), x + w / 2, y + cellH / 2 + 1);
        ctx.textAlign = "start";
      };
      cell(rowsY[0], ax.a, ax.a === null, ax.stretchA, ax.bad, ax.out ?? 0);
      cell(rowsY[1], ax.b, ax.b === null, ax.stretchB, ax.bad, ax.out ?? 0);
      // 결과 줄
      box(x, rowsY[2], w, cellH, 7);
      ctx.fillStyle = ax.bad ? alpha(C.edge, 0.3) : alpha(C.ink2, 0.28); ctx.fill();
      ctx.strokeStyle = ax.bad ? C.edge : C.ink2; ctx.lineWidth = ax.bad ? 2 : 1.2; ctx.stroke();
      ctx.font = "700 12px JetBrains Mono, monospace"; ctx.textAlign = "center";
      ctx.fillStyle = ax.bad ? C.edge : C.ink;
      ctx.fillText(ax.bad ? `${ax.ap} ≠ ${ax.bp}` : String(ax.out), x + w / 2, rowsY[2] + cellH / 2 + 1);
      ctx.textAlign = "start";
    });

    // 2) 칸 그림: 결과 shape 위에 a, b를 놓는다. 실제 원소는 진하게, 늘어난(stride 0) 칸은 점선으로
    const gy0 = rowsY[2] + cellH + 30, gy1 = H - 12;
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.mid;
    ctx.fillText(r.shape ? "원소 칸: 점선은 늘어난 칸(복사 없음)" : "원소 칸", pad, gy0 - 12);
    const shapes = r.shape ? [r.shape, r.shape, r.shape] : [r.axes.map((x) => x.ap), r.axes.map((x) => x.bp), null];
    const slotW = (W - 2 * pad - 2 * 26) / 3;
    const ops = [pr.op, "="];
    for (let g = 0; g < 3; g++) {
      const sx = pad + g * (slotW + 26);
      if (g < 2) { ctx.font = "700 18px JetBrains Mono, monospace"; ctx.fillStyle = C.mid; ctx.textAlign = "center"; ctx.fillText(ops[g], sx + slotW + 13, (gy0 + gy1) / 2); ctx.textAlign = "start"; }
      const sh = shapes[g];
      if (!sh) {
        ctx.font = "700 13px JetBrains Mono, monospace";
        ctx.fillStyle = C.edge; ctx.textAlign = "center";
        ctx.fillText("ValueError", sx + slotW / 2, (gy0 + gy1) / 2 - 8);
        ctx.font = "11px JetBrains Mono, monospace";
        ctx.fillText(`축 ${r.badAxis}`, sx + slotW / 2, (gy0 + gy1) / 2 + 10);
        ctx.textAlign = "start";
        continue;
      }
      // 칸이 '실제 원소'인지: 결과 칸 index에서, 그 입력이 늘어난 축의 index가 0이 아니면 가상 칸
      const realAt = (idx: number[]) => {
        if (!r.shape || g === 2) return true;
        return r.axes.every((ax, i) => !((g === 0 ? ax.ap : ax.bp) === 1 && idx[i] > 0));
      };
      drawArray(sx, gy0, slotW, gy1 - gy0, sh, realAt, g === 2 ? C.ink2 : g === 0 ? C.ink2 : C.mid);
    }
    ctx.textBaseline = "alphabetic";
  }

  /** 3차원 이하 배열을 칸으로. 3차원은 첫 축을 겹친 층으로 그린다. */
  function drawArray(x: number, y: number, w: number, h: number, sh: number[], realAt: (idx: number[]) => boolean, col: string) {
    const ctx = f.ctx;
    const s3 = sh.length === 3 ? sh : sh.length === 2 ? [1, ...sh] : [1, 1, ...sh];
    const [L, R, Cc] = s3;
    const off = Math.min(10, 44 / Math.max(L - 1, 1));
    const cell = Math.max(5, Math.min(26, (w - off * (L - 1)) / Cc, (h - off * (L - 1)) / R));
    const gw = Cc * cell + off * (L - 1), gh = R * cell + off * (L - 1);
    const bx = x + (w - gw) / 2, by = y + (h - gh) / 2 + off * (L - 1);
    for (let l = L - 1; l >= 0; l--) {
      const ox = bx + l * off, oy = by - l * off;
      for (let i = 0; i < R; i++) for (let j = 0; j < Cc; j++) {
        const idx3 = [l, i, j];
        const idx = idx3.slice(3 - sh.length);
        const real = realAt(idx);
        const cx = ox + j * cell, cy = oy + i * cell;
        // 층이 겹쳐도 뒤 층이 비치지 않도록 바탕을 먼저 칠한다
        ctx.fillStyle = C.bg;
        ctx.fillRect(cx, cy, cell, cell);
        if (real) { ctx.fillStyle = alpha(col, 0.42); ctx.fillRect(cx + 1, cy + 1, cell - 2, cell - 2); }
        ctx.setLineDash(real ? [] : [2, 2]);
        ctx.strokeStyle = real ? col : alpha(col, 0.7);
        ctx.lineWidth = 1;
        ctx.strokeRect(cx + 1.5, cy + 1.5, cell - 3, cell - 3);
        ctx.setLineDash([]);
      }
    }
  }
  function box(x: number, y: number, w: number, h: number, r: number) {
    const c = f.ctx; c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
  }

  const pre = presets(Object.entries(PRESETS).map(([k, p]) => ({ key: k, label: p.label })), (k) => { key = k; update(); }, key);
  pre.row.setAttribute("aria-label", "두 배열의 shape");
  const sliders: Record<Sym, ReturnType<typeof param>> = {
    N: param({ label: "N", min: 1, max: 6, step: 1, value: val.N, fmt: (v) => String(v), aria: "축 길이 N", onInput: (v) => { val.N = v; update(); } }),
    D: param({ label: "D", min: 1, max: 5, step: 1, value: val.D, fmt: (v) => String(v), aria: "축 길이 D", onInput: (v) => { val.D = v; update(); } }),
    M: param({ label: "M", min: 1, max: 6, step: 1, value: val.M, fmt: (v) => String(v), aria: "축 길이 M", onInput: (v) => { val.M = v; update(); } }),
  };
  const grid = el("div", "param-grid");
  (Object.keys(sliders) as Sym[]).forEach((s) => grid.append(sliders[s].row));
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);

  function update() {
    const pr = PRESETS[key];
    const used = new Set([...pr.a, ...pr.b].filter((d): d is Sym => d !== 1));
    (Object.keys(sliders) as Sym[]).forEach((s) => (sliders[s].row.hidden = !used.has(s)));
    const A = dims(pr.a), Bs = dims(pr.b), r = broadcast(A, Bs);
    const size = (s: number[]) => s.reduce((p, v) => p * v, 1);
    ro.set("a", `a ${fmtShape(A)}`);
    ro.set("b", `b ${fmtShape(Bs)}`);
    if (r.shape) {
      const big = size(r.shape) > Math.max(size(A), size(Bs));
      ro.set("out", `결과 ${fmtShape(r.shape)}`, "hot");
      ro.set("n", `원소 ${size(r.shape)}개${big ? " · 두 입력보다 크다" : ""}`, big ? "hot" : "");
    } else {
      const ax = r.axes[r.axes.length + r.badAxis];
      ro.set("out", `ValueError: 축 ${r.badAxis}에서 ${ax.ap} ≠ ${ax.bp}`, "hot");
      ro.set("n", "결과 없음");
    }
    draw();
  }
  f.onResize(draw);
  update();
}
