// 메모리 배치와 캐시 라인 (data/widgets.toml: strides-cache).
// float64 배열 A (6, 8)을 C order 또는 F order로 메모리에 놓고, 행 순회(j가 안쪽) 또는 열 순회(i가 안쪽)로 모든 원소를 한 번씩 읽는다.
// '전치 뷰'를 켜면 복사 없이 strides만 바꾼 A.T (8, 6)을 같은 방식으로 훑는다.
// 캐시 모델(단순화한 가상 모델): 캐시 라인 64 byte(원소 8개), 캐시에 라인 4개, 가장 오래 안 쓴 라인을 내보낸다(LRU), 처음엔 비어 있다.
// 배열(라인 6개)이 캐시(라인 4개)보다 크므로, 같은 라인을 다시 쓰기 전에 내보내면 매번 미스가 난다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

export const ROWS = 6, COLS = 8, ITEM = 8, LINE = 64, CACHE_LINES = 4;
export type Order = "C" | "F";
export type Walk = "row" | "col";

/** 보는 배열의 shape·strides(byte)와 순회 순서대로의 (i, j)·주소, 접근마다 적중 여부 */
export function simulate(order: Order, walk: Walk, transpose: boolean) {
  const base: [number, number] = order === "C" ? [COLS * ITEM, ITEM] : [ITEM, ROWS * ITEM];
  const shape: [number, number] = transpose ? [COLS, ROWS] : [ROWS, COLS];
  const strides: [number, number] = transpose ? [base[1], base[0]] : base;
  const seq: { i: number; j: number; addr: number; hit: boolean; cache: number[] }[] = [];
  const cache: number[] = []; // 앞쪽이 가장 최근
  const visit = (i: number, j: number) => {
    const addr = i * strides[0] + j * strides[1], line = Math.floor(addr / LINE);
    const at = cache.indexOf(line), hit = at >= 0;
    if (hit) cache.splice(at, 1); else if (cache.length === CACHE_LINES) cache.pop();
    cache.unshift(line);
    seq.push({ i, j, addr, hit, cache: [...cache] });
  };
  if (walk === "row") { for (let i = 0; i < shape[0]; i++) for (let j = 0; j < shape[1]; j++) visit(i, j); }
  else { for (let j = 0; j < shape[1]; j++) for (let i = 0; i < shape[0]; i++) visit(i, j); }
  const misses = seq.filter((s) => !s.hit).length;
  const inner = walk === "row" ? strides[1] : strides[0];
  return { shape, strides, seq, misses, hitRate: 1 - misses / seq.length, inner };
}

export function mount(root: HTMLElement) {
  const f = frame(root, "메모리 배치와 캐시 라인", "manipulate", (root.clientWidth || 600) < 480 ? 1.45 : 0.56, "배열의 논리 격자와 메모리 주소 순서를 캐시 라인 단위로 그리고, 순회 순서대로 캐시 적중과 미스를 표시한 그림");
  let order: Order = "C", walk: Walk = "col", transpose = false, t = ROWS * COLS;
  const N = ROWS * COLS;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const sim = simulate(order, walk, transpose);
    const pad = 14, narrow = W < 480;
    // 접근 번호: 칸마다 몇 번째로 읽혔고 적중이었는가
    const visitNo = new Map<string, number>();
    sim.seq.forEach((s, k) => visitNo.set(`${s.i},${s.j}`, k));
    const cur = t > 0 ? sim.seq[t - 1] : null;
    // 영역 나누기
    const L = narrow ? { gx: pad, gy: 30, gw: W - 2 * pad, gh: H * 0.48 - 30, mx: pad, my: H * 0.48 + 30, mw: W - 2 * pad, mh: H * 0.52 - 44 }
      : { gx: pad, gy: 30, gw: W * 0.42 - pad, gh: H - 44, mx: W * 0.47, my: 30, mw: W * 0.53 - pad, mh: H - 44 };
    ctx.textBaseline = "middle";
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.mid;
    ctx.fillText(transpose ? `A.T  (${sim.shape[0]}, ${sim.shape[1]})` : `A  (${sim.shape[0]}, ${sim.shape[1]})`, L.gx, L.gy - 16);
    ctx.fillText("메모리(주소 순서) · 한 줄 = 캐시 라인 64 B", L.mx, L.my - 16);

    const tone = (k: number) => (k >= t ? null : sim.seq[k].hit ? "hit" : "miss");
    const paint = (x: number, y: number, s: number, k: number) => {
      const st = tone(k);
      ctx.fillStyle = st === "miss" ? alpha(C.edge, 0.85) : st === "hit" ? alpha(C.ink2, 0.22) : C.bg;
      ctx.fillRect(x + 1, y + 1, s - 2, s - 2);
      ctx.strokeStyle = cur && k === t - 1 ? C.ink : C.faint;
      ctx.lineWidth = cur && k === t - 1 ? 2 : 1;
      ctx.strokeRect(x + 1.5, y + 1.5, s - 3, s - 3);
      if (s >= 17 && k < t) {
        ctx.font = `${s >= 24 ? 11 : 9}px JetBrains Mono, monospace`;
        ctx.fillStyle = st === "miss" ? C.bg : C.ink; ctx.textAlign = "center";
        ctx.fillText(String(k + 1), x + s / 2, y + s / 2 + 0.5);
        ctx.textAlign = "start";
      }
    };
    // 논리 격자
    const [R, Cc] = sim.shape;
    const gs = Math.min(36, (L.gw - 20) / Cc, (L.gh - 20) / R);
    const gx0 = L.gx + 18, gy0 = L.gy + 14;
    ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = C.mid; ctx.textAlign = "center";
    for (let j = 0; j < Cc; j++) ctx.fillText(String(j), gx0 + (j + 0.5) * gs, gy0 - 8);
    for (let i = 0; i < R; i++) ctx.fillText(String(i), gx0 - 10, gy0 + (i + 0.5) * gs);
    ctx.textAlign = "start";
    for (let i = 0; i < R; i++) for (let j = 0; j < Cc; j++) paint(gx0 + j * gs, gy0 + i * gs, gs, visitNo.get(`${i},${j}`)!);

    // 메모리: 주소 0..383을 캐시 라인(64 B = 원소 8개)마다 한 줄로
    const nLines = (N * ITEM) / LINE, per = LINE / ITEM;
    const ms = Math.min(34, (L.mw - 44) / per, (L.mh - 64) / nLines);
    const mx0 = L.mx + 34, my0 = L.my + 4;
    const byAddr = new Map<number, number>();
    sim.seq.forEach((s, k) => byAddr.set(s.addr, k));
    const inCache = new Set(cur ? cur.cache : []);
    for (let ln = 0; ln < nLines; ln++) {
      const y = my0 + ln * ms;
      ctx.font = "10px JetBrains Mono, monospace";
      ctx.fillStyle = inCache.has(ln) ? C.edge : C.mid;
      ctx.textAlign = "right";
      ctx.fillText(`L${ln}`, mx0 - 8, y + ms / 2);
      ctx.textAlign = "start";
      for (let e = 0; e < per; e++) paint(mx0 + e * ms, y, ms, byAddr.get((ln * per + e) * ITEM)!);
      // 지금 캐시에 있는 라인은 테두리로 표시
      if (inCache.has(ln)) { ctx.strokeStyle = C.edge; ctx.lineWidth = 2; ctx.strokeRect(mx0 - 1, y, per * ms + 2, ms); }
    }
    // 캐시 상태
    const cy = my0 + nLines * ms + 22;
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
    ctx.fillText("캐시(라인 4개)", L.mx, cy);
    const slotW = Math.min(40, (L.mw - 110) / CACHE_LINES);
    for (let c = 0; c < CACHE_LINES; c++) {
      const x = L.mx + 100 + c * (slotW + 6);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x, cy - 11, slotW, 22);
      const ln = cur?.cache[c];
      if (ln !== undefined) {
        ctx.fillStyle = alpha(C.edge, c === 0 ? 0.3 : 0.12); ctx.fillRect(x + 1, cy - 10, slotW - 2, 20);
        ctx.font = "600 11px JetBrains Mono, monospace"; ctx.fillStyle = C.ink; ctx.textAlign = "center";
        ctx.fillText(`L${ln}`, x + slotW / 2, cy + 1); ctx.textAlign = "start";
      }
    }
    ctx.textBaseline = "alphabetic";
  }

  const ordBtns = presets<Order>([{ key: "C", label: "C order" }, { key: "F", label: "F order" }], (k) => { order = k; reset(); }, order);
  ordBtns.row.setAttribute("aria-label", "메모리 배치");
  const walkBtns = presets<Walk>([{ key: "row", label: "행 순회 (j 안쪽)" }, { key: "col", label: "열 순회 (i 안쪽)" }], (k) => { walk = k; reset(); }, walk);
  walkBtns.row.setAttribute("aria-label", "순회 순서");
  const tools = el("div", "wframe-row");
  const tog = el("label", "toggle");
  tog.innerHTML = `<input type="checkbox" /> 전치 뷰 A.T`;
  tog.querySelector("input")!.addEventListener("change", (e) => { transpose = (e.target as HTMLInputElement).checked; reset(); });
  tools.append(tog);
  const pt = param({ label: "접근", min: 0, max: N, step: 1, value: t, fmt: (v) => `${v}/${N}`, aria: "몇 번째 접근까지 보기", onInput: (v) => { t = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pt.row);
  const ro = readouts();
  f.panel.append(ordBtns.row, walkBtns.row, tools, grid, ro.row);

  function reset() { t = N; pt.set(t); update(); }
  function update() {
    const sim = simulate(order, walk, transpose);
    const seen = sim.seq.slice(0, t), miss = seen.filter((s) => !s.hit).length;
    ro.set("strides", `strides (${sim.strides[0]}, ${sim.strides[1]}) B`);
    ro.set("inner", `안쪽 반복의 보폭 ${sim.inner} B`, sim.inner === ITEM ? "ok" : "hot");
    ro.set("miss", `접근 ${t} · 미스 ${miss}`);
    ro.set("rate", `적중률 ${t ? (((t - miss) / t) * 100).toFixed(1) : "0.0"}%`, t && (t - miss) / t > 0.8 ? "ok" : "hot");
    draw();
  }
  f.onResize(draw);
  update();
}
