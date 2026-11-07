// 수용 영역 (data/widgets.toml: receptive-field).
// 18.2장 "receptive field": 층 l의 kernel k_l, stride s_l, dilation d_l로 쌓은 1차원 CNN에서
// 이론 RF r = 1 + Σ (k_l − 1)·d_l·Π_{i<l} s_i를 계산하고, 맨 위 출력 한 칸이 각 층의 어느 칸에 닿는지를 격자에 표시한다.
// 칸의 진하기는 모든 원소가 1/k인 kernel로 본 영향(출력에서 그 칸까지 경로 가중치의 합)이다.
// 본문 셀처럼 영향의 95%가 담긴 가운데 폭(유효 RF)을 함께 센다. kernel은 출력 위치를 가운데로 두므로 k는 홀수만 쓴다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

export type Layer = { k: number; s: number; d: number };
const DIL = [1, 2, 4, 8];
const MAXL = 6;

export function theoreticalRF(layers: Layer[]) {
  let r = 1, jump = 1;
  for (const { k, s, d } of layers) { r += (k - 1) * d * jump; jump *= s; }
  return { r, jump };
}

/** 위에서부터 영향을 내려보낸다. maps[l]은 층 l(0 = 입력)의 칸 번호 → 영향 */
export function influence(layers: Layer[]) {
  const maps: Map<number, number>[] = new Array(layers.length + 1);
  let cur = new Map<number, number>([[0, 1]]);
  maps[layers.length] = cur;
  for (let l = layers.length - 1; l >= 0; l--) {
    const { k, s, d } = layers[l], c = Math.floor((k - 1) / 2);
    const next = new Map<number, number>();
    for (const [m, v] of cur) for (let u = 0; u < k; u++) { const i = m * s + (u - c) * d; next.set(i, (next.get(i) ?? 0) + v / k); }
    maps[l] = cur = next;
  }
  return maps;
}

/** 출력 위치(0)를 가운데로, 영향의 95%가 담기는 가장 좁은 대칭 창의 폭 */
export function width95(m0: Map<number, number>) {
  const R = Math.max(...[...m0.keys()].map(Math.abs));
  let acc = m0.get(0) ?? 0;
  for (let m = 0; m <= R; m++) {
    if (m > 0) acc += (m0.get(m) ?? 0) + (m0.get(-m) ?? 0);
    if (acc >= 0.95 - 1e-12) return 2 * m + 1;
  }
  return 2 * R + 1;
}

const PRESETS = {
  plain: { label: "3 × 5층", layers: Array.from({ length: 5 }, () => ({ k: 3, s: 1, d: 1 })) },
  stride: { label: "stride 2", layers: Array.from({ length: 5 }, () => ({ k: 3, s: 2, d: 1 })) },
  dil: { label: "d 1·2·4·8", layers: [1, 2, 4, 8].map((d) => ({ k: 3, s: 1, d })) },
  dil2: { label: "d 2 반복", layers: Array.from({ length: 5 }, () => ({ k: 3, s: 1, d: 2 })) },
} as const;
type PK = keyof typeof PRESETS;

export function mount(root: HTMLElement) {
  const f = frame(root, "수용 영역", "manipulate", 0.5, "층마다 출력 한 칸이 닿는 칸을 표시한 1차원 합성곱 격자");
  let stack: Layer[] = Array.from({ length: MAXL }, () => ({ k: 3, s: 1, d: 1 }));
  let L = 5, sel = L - 1;
  let maps = influence(stack.slice(0, L));

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const layers = stack.slice(0, L);
    const PL = 44, PR = 10, top = 14, bottom = H - 26;
    const keys0 = [...maps[0].keys()];
    const lo = Math.min(...keys0), hi = Math.max(...keys0);
    const pad = Math.max(2, Math.round((hi - lo + 1) * 0.06));
    const a = lo - pad, b = hi + pad;
    const px = (W - PL - PR) / (b - a + 1);
    const X = (x: number) => PL + (x - a + 0.5) * px;
    const rowH = (bottom - top) / (L + 1);
    const Yr = (l: number) => bottom - rowH * (l + 0.5);
    const jumps = [1];
    layers.forEach(({ s }, i) => jumps.push(jumps[i] * s));
    // 이론 RF와 유효 RF 괄호(입력 줄 아래)
    const w95 = width95(maps[0]);
    const half95 = (w95 - 1) / 2;
    const yb = bottom + 6;
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(X(lo) - px / 2, yb - 4); ctx.lineTo(X(lo) - px / 2, yb); ctx.lineTo(X(hi) + px / 2, yb); ctx.lineTo(X(hi) + px / 2, yb - 4); ctx.stroke();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(X(-half95) - px / 2, yb + 4); ctx.lineTo(X(half95) + px / 2, yb + 4); ctx.stroke();
    for (let l = 0; l <= L; l++) {
      const y = Yr(l), J = jumps[l], h = Math.min(rowH * 0.56, 22);
      const isSel = l === sel + 1;
      ctx.fillStyle = isSel ? C.ink : C.mid;
      ctx.font = isSel ? "600 11px JetBrains Mono, monospace" : "11px JetBrains Mono, monospace";
      ctx.fillText(l === 0 ? "입력" : `층 ${l}`, 4, y + 4);
      ctx.font = "11px JetBrains Mono, monospace";
      const cellPx = J * px;
      const m = maps[l];
      let vmax = 0; for (const v of m.values()) vmax = Math.max(vmax, v);
      if (cellPx >= 3) {
        const w = Math.max(1.5, Math.min(cellPx - 1.2, 18));
        // 이 층의 모든 칸(입력 좌표 J 간격)
        ctx.fillStyle = alpha(C.mid, 0.16);
        for (let i = Math.ceil(a / J); i * J <= b; i++) ctx.fillRect(X(i * J) - w / 2, y - h / 2, w, h);
        for (const [i, v] of m) {
          ctx.fillStyle = alpha(C.edge, 0.25 + 0.75 * (v / vmax));
          ctx.fillRect(X(i * J) - w / 2, y - h / 2, w, h);
        }
      } else {
        // 칸이 화면 1–2 px보다 좁으면 픽셀 열마다 최댓값으로 모은다
        ctx.fillStyle = alpha(C.mid, 0.16);
        ctx.fillRect(PL, y - h / 2, W - PL - PR, h);
        const cols = new Float64Array(Math.ceil(W));
        for (const [i, v] of m) { const c = Math.floor(X(i * J)); if (c >= 0 && c < cols.length) cols[c] = Math.max(cols[c], v); }
        cols.forEach((v, c) => { if (!v) return; ctx.fillStyle = alpha(C.edge, 0.25 + 0.75 * (v / vmax)); ctx.fillRect(c, y - h / 2, 1, h); });
      }
      // 아래 층으로 가는 연결선(칸이 적을 때만)
      if (l > 0 && m.size * layers[l - 1].k <= 160) {
        const { k, s, d } = layers[l - 1], c = Math.floor((k - 1) / 2), Jd = jumps[l - 1];
        ctx.strokeStyle = alpha(C.edge, 0.35); ctx.lineWidth = 1;
        ctx.beginPath();
        for (const i of m.keys()) for (let u = 0; u < k; u++) {
          const j = i * s + (u - c) * d;
          ctx.moveTo(X(i * J), y + h / 2); ctx.lineTo(X(j * Jd), Yr(l - 1) - h / 2);
        }
        ctx.stroke();
      }
    }
  }

  const layerRow = el("div", "wframe-row");
  layerRow.setAttribute("role", "group");
  layerRow.setAttribute("aria-label", "고칠 층");
  const layerBtns: HTMLButtonElement[] = [];
  for (let i = 0; i < MAXL; i++) {
    const btn = el("button", "preset");
    btn.type = "button";
    btn.textContent = `층 ${i + 1}`;
    btn.addEventListener("click", () => { sel = i; syncSliders(); update(); });
    layerRow.append(btn); layerBtns.push(btn);
  }

  function syncSliders() {
    const { k, s, d } = stack[sel];
    pk.set(k); ps.set(s); pd.set(DIL.indexOf(d));
    layerBtns.forEach((btn, i) => { btn.hidden = i >= L; btn.setAttribute("aria-pressed", String(i === sel)); });
  }

  let raf = 0;
  function update() {
    const layers = stack.slice(0, L);
    maps = influence(layers);
    const { r, jump } = theoreticalRF(layers);
    const reach = maps[0].size;
    ro.set("r", `이론 RF ${r}`, "hot");
    ro.set("reach", reach < r ? `닿는 입력 ${reach} (구멍 ${r - reach})` : `닿는 입력 ${reach}`);
    ro.set("w95", `95% 폭 ${width95(maps[0])}`);
    ro.set("res", `해상도 1/${jump}`);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const edit = (fn: (ly: Layer) => void) => { fn(stack[sel]); pre.select(null); update(); };
  const pre = presets((Object.keys(PRESETS) as PK[]).map((key) => ({ key, label: PRESETS[key].label })), (key) => {
    const ls = PRESETS[key].layers;
    L = ls.length;
    stack = Array.from({ length: MAXL }, (_, i) => ({ ...(ls[i] ?? { k: 3, s: 1, d: 1 }) }));
    sel = Math.min(sel, L - 1);
    pL.set(L); syncSliders(); update();
  }, "plain");
  const pL = (param({ label: "L", min: 1, max: MAXL, step: 1, value: L, fmt: (v) => `${v}`, aria: "층 수 L", onInput: (v) => { L = v; sel = Math.min(sel, L - 1); pre.select(null); syncSliders(); update(); } }));
  const pk = (param({ label: "k", min: 1, max: 7, step: 2, value: 3, fmt: (v) => `${v}`, aria: "고른 층의 kernel 크기", onInput: (v) => edit((ly) => (ly.k = v)) }));
  const ps = (param({ label: "s", min: 1, max: 3, step: 1, value: 1, fmt: (v) => `${v}`, aria: "고른 층의 stride", onInput: (v) => edit((ly) => (ly.s = v)) }));
  const pd = (param({ label: "d", min: 0, max: DIL.length - 1, step: 1, value: 0, fmt: (v) => `${DIL[v]}`, aria: "고른 층의 dilation", onInput: (v) => edit((ly) => (ly.d = DIL[v])) }));
  const grid = el("div", "param-grid");
  grid.append(pL.row, pk.row, ps.row, pd.row);
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(140px, 1fr))";   // 수업 시트 폭에서도 두 칸
  const ro = readouts();
  f.panel.append(pre.row, layerRow, grid, ro.row);
  syncSliders();
  f.onResize(draw);
  update();
}
