// einsum 축 표기 (data/widgets.toml: einsum-axes, 7.7장 "인덱스 표기와 einsum").
// 식 하나를 고르면 입력과 출력을 축 칸 그림으로 그리고, 출력 원소 하나를 고르면 그 값에 곱해져 더해지는 입력 칸을 강조한다.
// 규칙(본문): 여러 입력에 같은 글자가 있으면 같은 위치끼리 곱하고, 출력에 없는 글자는 합해서 없앤다. 한 입력 안의 반복 글자는 대각만 읽는다.
// 축 크기는 글자마다 모두 다르게 잡는다(b=2, i=3, j=4, k=5, t=3, s=4, d=2 — 본문의 "테스트에서 축 크기를 모두 다르게").
// 입력은 np.arange처럼 0, 1, 2, …로 채운다. 3축 텐서는 첫 축(batch)마다 한 장씩 위아래로 쌓는다.
import { frame, param, presets, readouts, el, C } from "../frame";

export const SIZES: Record<string, number> = { b: 2, i: 3, j: 4, k: 5, t: 3, s: 4, d: 2 };

export interface Spec { ins: string[]; out: string }
export const parse = (expr: string): Spec => { const [l, out] = expr.split("->"); return { ins: l.split(","), out }; };

const shapeOf = (axes: string) => [...axes].map((c) => SIZES[c]);
const sizeOf = (axes: string) => shapeOf(axes).reduce((a, b) => a * b, 1);
/** C 순서 평탄 위치 */
const flat = (axes: string, idx: Record<string, number>) => [...axes].reduce((o, c) => o * SIZES[c] + idx[c], 0);

/** 모든 글자의 값 조합을 돌며 einsum을 직접 계산한다(입력 k번째는 0, 1, 2, … 채움). */
export function einsum(spec: Spec) {
  const letters = [...new Set(spec.ins.join("") + spec.out)];
  const out = new Array(sizeOf(spec.out)).fill(0);
  let mults = 0;
  const idx: Record<string, number> = {};
  const rec = (n: number) => {
    if (n === letters.length) {
      let prod = 1;
      for (const a of spec.ins) prod *= flat(a, idx);
      out[flat(spec.out, idx)] += prod;
      mults++;
      return;
    }
    for (let v = 0; v < SIZES[letters[n]]; v++) { idx[letters[n]] = v; rec(n + 1); }
  };
  rec(0);
  return { out, mults, letters };
}
/** 출력 평탄 위치 o에 기여하는 (입력 번호 → 평탄 위치 집합)과 항 목록 */
export function contributors(spec: Spec, o: number) {
  const fixed: Record<string, number> = {};
  let rem = o;
  for (const c of [...spec.out].reverse()) { fixed[c] = rem % SIZES[c]; rem = Math.floor(rem / SIZES[c]); }
  const summed = [...new Set(spec.ins.join(""))].filter((c) => !spec.out.includes(c));
  const cells = spec.ins.map(() => new Set<number>());
  const terms: number[][] = [];
  const idx: Record<string, number> = { ...fixed };
  const rec = (n: number) => {
    if (n === summed.length) {
      const pos = spec.ins.map((a) => flat(a, idx));
      // 한 입력 안 반복 글자(ii)는 같은 값이어야 대각이다 — idx가 글자마다 하나라 자동으로 지켜진다
      pos.forEach((p, k) => cells[k].add(p));
      terms.push(pos);
      return;
    }
    for (let v = 0; v < SIZES[summed[n]]; v++) { idx[summed[n]] = v; rec(n + 1); }
  };
  rec(0);
  return { fixed, summed, cells, terms };
}

const PRESETS: { key: string; label: string; expr: string }[] = [
  { key: "matmul", label: "ij,jk->ik", expr: "ij,jk->ik" },
  { key: "bmm", label: "bij,bjk->bik", expr: "bij,bjk->bik" },
  { key: "attn", label: "btd,bsd->bts", expr: "btd,bsd->bts" },
  { key: "trace", label: "ii->", expr: "ii->" },
  { key: "diag", label: "ii->i", expr: "ii->i" },
  { key: "transpose", label: "ij->ji", expr: "ij->ji" },
  { key: "bdot", label: "bi,bi->b", expr: "bi,bi->b" },
  { key: "outer", label: "i,j->ij", expr: "i,j->ij" },
];
const NAMES = ["A", "B"];

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  // 가장 큰 식(bij,bjk->bik)이 들어가는 높이: 넓으면 한 줄, 좁으면 입력 / 출력 두 줄
  const f = frame(root, "einsum 축 표기", "manipulate", root.clientWidth < 480 ? 1.15 : 330 / Math.max(root.clientWidth, 300),
    "einsum 식의 입력과 출력 텐서를 칸으로 그리고, 고른 출력 원소에 기여하는 입력 칸을 강조한 그림");
  let cur = PRESETS.find((p) => p.key === attrs.preset) ?? PRESETS[1];
  let spec = parse(cur.expr);
  let sel = 0;
  let hits: { k: number; x: number; y: number; w: number; h: number; pos: number }[] = [];

  /** 피연산자 하나의 칸 배치: 3축이면 첫 축마다 2차원 판을 위아래로, 2축은 행×열, 1축은 한 줄, 0축은 한 칸 */
  const dims = (axes: string, cs: number) => {
    const sh = shapeOf(axes);
    const gap = cs * 0.5;
    if (sh.length === 0) return { w: cs, h: cs };
    if (sh.length === 1) return { w: sh[0] * cs, h: cs };
    if (sh.length === 2) return { w: sh[1] * cs, h: sh[0] * cs };
    return { w: sh[2] * cs, h: sh[0] * sh[1] * cs + (sh[0] - 1) * gap };
  };
  const cellRect = (axes: string, cs: number, ox: number, oy: number, pos: number) => {
    const sh = shapeOf(axes);
    const gap = cs * 0.5;
    if (sh.length === 0) return [ox, oy];
    if (sh.length === 1) return [ox + pos * cs, oy];
    if (sh.length === 2) return [ox + (pos % sh[1]) * cs, oy + Math.floor(pos / sh[1]) * cs];
    const per = sh[1] * sh[2];
    const s = Math.floor(pos / per), r = pos % per;
    return [ox + (r % sh[2]) * cs, oy + s * (sh[1] * cs + gap) + Math.floor(r / sh[2]) * cs];
  };

  function geom() {
    const W = f.width, H = f.height;
    const ops = [...spec.ins, spec.out];
    const sep = 44; // 쉼표·화살표 자리(3축 텐서의 판 번호도 이 자리에 쓴다)
    const labelH = 40, top = 44;
    // 한 줄 배치가 들어가는 가장 큰 칸 크기
    const oneRow = (cs: number) => ops.reduce((s, a) => s + dims(a, cs).w, 0) + sep * ops.length + 40;
    const maxH = (cs: number, list: string[]) => Math.max(...list.map((a) => dims(a, cs).h));
    const fits1 = (c: number) => oneRow(c) <= W && maxH(c, ops) + top + labelH <= H;
    // 두 줄: 입력 / 출력
    const fits2 = (c: number) => {
      const w1 = spec.ins.reduce((s, a) => s + dims(a, c).w, 0) + sep * spec.ins.length + 24;
      const h = maxH(c, spec.ins) + dims(spec.out, c).h + 2 * labelH + top;
      return w1 <= W && h <= H;
    };
    // 칸이 더 크게 들어가는 배치를 고른다(같으면 한 줄)
    let c1 = 34, c2 = 34;
    while (c1 > 12 && !fits1(c1)) c1--;
    while (c2 > 12 && !fits2(c2)) c2--;
    if (fits1(c1) && c1 >= c2) return { cs: c1, rows: [ops], sep, top, labelH };
    return { cs: c2, rows: [spec.ins, [spec.out]], sep, top, labelH };
  }

  function draw() {
    const { ctx, width: W } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, f.height);
    const { out } = einsum(spec);
    const con = contributors(spec, sel);
    const summed = new Set(con.summed);
    const batchLike = new Set([...spec.out].filter((c) => spec.ins.every((a) => a.includes(c)) && spec.ins.length > 1));
    // 식: 합하는 글자는 강조색, 모든 입력과 출력에 있는 글자(batch처럼 같이 도는 축)는 굵게
    ctx.font = "700 18px JetBrains Mono, monospace";
    const expr = cur.expr;
    let x = (W - ctx.measureText(expr).width) / 2;
    for (const ch of expr) {
      ctx.fillStyle = summed.has(ch) ? C.edge : batchLike.has(ch) ? C.ink : /[a-z]/.test(ch) ? C.ink : C.mid;
      ctx.fillText(ch, x, 26);
      if (summed.has(ch)) { ctx.fillRect(x, 30, ctx.measureText(ch).width, 2); }
      x += ctx.measureText(ch).width;
    }
    const g = geom();
    const { cs, sep, top, labelH } = g;
    hits = [];
    let y0 = top;
    g.rows.forEach((row) => {
      const widths = row.map((a) => dims(a, cs).w);
      const total = widths.reduce((s, w) => s + w, 0) + sep * (row.length - 1);
      let ox = (W - total) / 2;
      const rowH = Math.max(...row.map((a) => dims(a, cs).h));
      row.forEach((axes, ri) => {
        const isOut = row === g.rows[g.rows.length - 1] && ri === row.length - 1;
        const opIndex = isOut ? -1 : ri;
        const d = dims(axes, cs);
        const oy = y0 + 14;
        // 머리: 이름(shape)
        ctx.font = "600 12px JetBrains Mono, monospace";
        ctx.fillStyle = C.ink;
        const name = isOut ? "out" : NAMES[opIndex];
        const shp = `(${shapeOf(axes).join(",")}${shapeOf(axes).length === 1 ? "," : ""})`;
        ctx.fillText(`${name} ${shp}`, ox, y0 + 6);
        const n = sizeOf(axes);
        for (let pos = 0; pos < n; pos++) {
          const [cx, cy] = cellRect(axes, cs, ox, oy, pos);
          const hot = isOut ? pos === sel : con.cells[opIndex].has(pos);
          ctx.fillStyle = hot ? (isOut ? C.edge : C.fill) : C.bg;
          ctx.fillRect(cx, cy, cs, cs);
          ctx.strokeStyle = hot ? C.edge : C.faint;
          ctx.lineWidth = hot ? 1.6 : 1;
          ctx.strokeRect(cx + 0.5, cy + 0.5, cs - 1, cs - 1);
          const v = isOut ? out[pos] : pos;
          const txt = String(v);
          let fsz = Math.min(12, cs * 0.42);
          ctx.font = `${fsz}px JetBrains Mono, monospace`;
          const tw = ctx.measureText(txt).width;
          if (tw > cs - 4) { fsz = Math.max(7, (fsz * (cs - 4)) / tw); ctx.font = `${fsz}px JetBrains Mono, monospace`; }
          if (ctx.measureText(txt).width <= cs - 2) {
            ctx.fillStyle = hot && isOut ? C.bg : C.ink;
            ctx.fillText(txt, cx + (cs - ctx.measureText(txt).width) / 2, cy + cs / 2 + 4);
          }
          if (isOut) hits.push({ k: -1, x: cx, y: cy, w: cs, h: cs, pos });
        }
        // 축 이름: 행 축은 왼쪽, 열 축은 아래
        const sh = [...axes];
        ctx.font = "600 11px JetBrains Mono, monospace";
        const colAxis = sh[sh.length - 1], rowAxis = sh.length >= 2 ? sh[sh.length - 2] : null;
        if (colAxis) { ctx.fillStyle = summed.has(colAxis) ? C.edge : C.mid; ctx.fillText(`${colAxis} →`, ox, oy + d.h + 13); }
        if (rowAxis) { ctx.fillStyle = summed.has(rowAxis) ? C.edge : C.mid; ctx.fillText(`${rowAxis}↓`, ox - ctx.measureText(`${rowAxis}↓`).width - 3, oy + 11); }
        if (sh.length === 3) {
          ctx.fillStyle = summed.has(sh[0]) ? C.edge : C.mid;
          ctx.font = "10px JetBrains Mono, monospace";
          for (let s = 0; s < SIZES[sh[0]]; s++) ctx.fillText(`${sh[0]}=${s}`, ox + d.w + 3, oy + s * (SIZES[sh[1]] * cs + cs * 0.5) + 11);
        }
        ox += d.w;
        if (ri < row.length - 1) {
          ctx.fillStyle = C.mid; ctx.font = "700 16px JetBrains Mono, monospace";
          const isArrow = ri === row.length - 2 && g.rows.length === 1;
          ctx.fillText(isArrow ? "→" : ",", ox + 25, oy + Math.min(d.h, rowH) / 2 + 6);
          ox += sep;
        }
      });
      y0 += rowH + labelH;
    });
  }

  // 조작
  const pre = presets(PRESETS.map(({ key, label }) => ({ key, label })), (key) => {
    cur = PRESETS.find((p) => p.key === key)!; spec = parse(cur.expr); sel = 0; syncSel(); update();
  }, cur.key);
  const ps = param({ label: "원소", min: 0, max: 0, step: 1, value: 0, fmt: (v) => idxText(v), aria: "강조할 출력 원소의 위치", onInput: (v) => { sel = v; update(); } });
  ps.row.style.gridTemplateColumns = "2.4em minmax(0, 1fr) 7.2em";
  const grid = el("div", "param-grid");
  grid.append(ps.row);
  const ro = readouts();
  pre.row.style.flexWrap = "nowrap"; // 좁은 화면에서는 묶음 안에서 가로로 넘긴다(알약 모양 유지)
  f.panel.append(pre.row, grid, ro.row);

  function idxText(o: number) {
    if (!spec.out) return "out[()]";
    const { fixed } = contributors(spec, o);
    return `out[${[...spec.out].map((c) => fixed[c]).join(",")}]`;
  }
  function syncSel() {
    const n = sizeOf(spec.out);
    ps.input.max = String(n - 1);
    ps.input.disabled = n === 1;
    ps.set(sel);
  }
  function update() {
    const { out, mults } = einsum(spec);
    const con = contributors(spec, sel);
    const shp = shapeOf(spec.out);
    ro.set("shape", `결과 shape (${shp.join(", ")}${shp.length === 1 ? "," : ""})`, "hot");
    ro.set("sum", con.summed.length ? `합하는 축: ${con.summed.join(", ")}` : "합하는 축 없음");
    ro.set("mult", `곱셈 ${mults}번 = ${[...new Set(spec.ins.join("") + spec.out)].map((c) => SIZES[c]).join("·")}`);
    const ref = (k: number, p: number) => {
      const axes = spec.ins[k];
      let rem = p; const parts: number[] = [];
      for (const c of [...axes].reverse()) { parts.unshift(rem % SIZES[c]); rem = Math.floor(rem / SIZES[c]); }
      return `${NAMES[k]}[${parts.join(",")}]`;
    };
    const term = con.terms[0].map((p, k) => ref(k, p)).join("·");
    const sumTxt = con.summed.length ? `Σ_${con.summed.join("")} ` : "";
    ro.set("val", `${idxText(sel)} = ${sumTxt}${term}${con.terms.length > 1 ? " + …" : ""} = ${out[sel]} (${con.terms.length}항)`);
    draw();
  }

  f.canvas.addEventListener("pointerdown", (e) => {
    const r = f.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const h = hits.find((c) => x >= c.x && x < c.x + c.w && y >= c.y && y < c.y + c.h);
    if (h) { sel = h.pos; ps.set(sel); update(); }
  });
  f.canvas.addEventListener("pointermove", (e) => {
    const r = f.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    f.canvas.style.cursor = hits.some((c) => x >= c.x && x < c.x + c.w && y >= c.y && y < c.y + c.h) ? "pointer" : "default";
  });

  syncSel();
  f.onResize(draw);
  update();
}
