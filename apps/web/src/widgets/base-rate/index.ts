// 기저율과 자연 빈도 (data/widgets.toml: base-rate).
// 유병률 p, 민감도 P(+|병), 위양성률 P(+|정상)로 1000명을 자연 빈도로 나눈다(11.3장의 진단 검사 예제).
// 환자 = round(1000p), 참양성 = round(환자 × 민감도), 위양성 = round(정상 × 위양성률). 격자에서 환자는 앞쪽, 양성은 한 덩어리로 놓는다.
// 오른쪽(아래) 막대: 양성인 사람 중 환자의 비율. readouts의 P(병|+)는 베이즈 정리로 계산한 정확한 값이다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";

const N = 1000;
export function counts(prev: number, sens: number, fpr: number) {
  const sick = Math.round(N * prev), healthy = N - sick;
  const tp = Math.round(sick * sens), fp = Math.round(healthy * fpr);
  return { sick, healthy, tp, fn: sick - tp, fp, tn: healthy - fp };
}
export const posterior = (prev: number, sens: number, fpr: number) => (sens * prev) / (sens * prev + fpr * (1 - prev));

const PREVS = [
  { key: "p001", label: "0.1%", v: 0.001 }, { key: "p01", label: "1%", v: 0.01 },
  { key: "p10", label: "10%", v: 0.1 }, { key: "p50", label: "50%", v: 0.5 },
] as const;
type PK = (typeof PREVS)[number]["key"];

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 480;
  const f = frame(root, "기저율과 자연 빈도", "manipulate", narrow ? 0.86 : 0.56, "1000명을 환자·정상과 검사 양성·음성으로 나눈 격자, 그리고 양성 중 환자의 비율 막대");
  let prev = 0.001, sens = 0.99, fpr = 0.05;
  let k = counts(prev, sens, fpr);
  // 칸의 순서: 위음성 → 참양성 → 위양성 → 참음성 (환자와 양성이 각각 이어진 덩어리가 된다)
  const kind = (i: number) => (i < k.fn ? "fn" : i < k.fn + k.tp ? "tp" : i < k.fn + k.tp + k.fp ? "fp" : "tn");

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const wide = W >= 480;
    // 위양성은 보조 색을 흐리게 써서 참양성(주된 색)과 테마에 상관없이 구별되게 한다
    const fpCol = alpha(C.ink2, 0.55);
    const cols = 40, rows = N / cols;
    const barW = wide ? 120 : W - 24, legendH = 22;
    const gw = wide ? W - barW - 44 : W - 24, gh = wide ? H - 24 - legendH : H - 24 - legendH - 86;
    const cell = Math.min(gw / cols, gh / rows);
    const gx = 12 + (gw - cell * cols) / 2, gy = 12;
    const r = Math.max(1.2, cell * 0.36);
    for (let i = 0; i < N; i++) {
      const cx = gx + (i % cols) * cell + cell / 2, cy = gy + Math.floor(i / cols) * cell + cell / 2;
      const t = kind(i);
      ctx.beginPath(); ctx.arc(cx, cy, t === "tn" ? r * 0.75 : r, 0, 7);
      if (t === "tp") { ctx.fillStyle = C.edge; ctx.fill(); }
      else if (t === "fp") { ctx.fillStyle = fpCol; ctx.fill(); }
      else if (t === "fn") { ctx.strokeStyle = C.edge; ctx.lineWidth = 1.4; ctx.stroke(); }
      else { ctx.fillStyle = C.faint; ctx.fill(); }
    }
    // 범례(폭이 모자라면 줄을 바꾼다)
    ctx.font = "11px JetBrains Mono, monospace";
    const items: [string, string, boolean][] = [[`참양성 ${k.tp}`, C.edge, true], [`위양성 ${k.fp}`, fpCol, true], [`위음성 ${k.fn}`, C.edge, false], [`참음성 ${k.tn}`, C.faint, true]];
    let lx = gx, ly = gy + rows * cell + 16;
    const right = wide ? gx + cols * cell : W - 12;
    for (const [txt, col, fill] of items) {
      const w = ctx.measureText(txt).width + 14;
      if (lx + w > right && lx > gx) { lx = gx; ly += 16; }
      ctx.beginPath(); ctx.arc(lx + 5, ly - 4, 4.5, 0, 7);
      if (fill) { ctx.fillStyle = col; ctx.fill(); } else { ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.stroke(); }
      ctx.fillStyle = C.ink; ctx.fillText(txt, lx + 14, ly);
      lx += w + 16;
    }
    // 양성 중 환자 비율 막대(코랄 = 환자, 파랑 = 위양성)
    const pos = k.tp + k.fp;
    const share = pos ? k.tp / pos : 0;
    const lbl = `환자 ${k.tp} / ${pos}`, pct = `= ${(share * 100).toFixed(1)}%`;
    ctx.fillStyle = C.mid;
    if (wide) {
      const bx = W - barW - 12 + 20, by = 30, bw = 40, bh = H - 92;
      ctx.fillText(`양성 ${pos}명`, bx - 8, by - 10);
      ctx.fillStyle = fpCol; ctx.fillRect(bx, by, bw, bh);
      ctx.fillStyle = C.edge; ctx.fillRect(bx, by + bh * (1 - share), bw, bh * share);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(bx, by, bw, bh);
      ctx.fillStyle = C.ink; ctx.font = "600 12px JetBrains Mono, monospace";
      ctx.fillText(lbl, bx - 8, by + bh + 18); ctx.fillText(pct, bx - 8, by + bh + 34);
    } else {
      const bx = 12, by = ly + 30, bw = W - 24, bh = 20;
      ctx.fillText(`양성 ${pos}명`, bx, by - 8);
      ctx.fillStyle = fpCol; ctx.fillRect(bx, by, bw, bh);
      ctx.fillStyle = C.edge; ctx.fillRect(bx, by, bw * share, bh);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(bx, by, bw, bh);
      ctx.fillStyle = C.ink; ctx.font = "600 12px JetBrains Mono, monospace";
      ctx.fillText(`${lbl} ${pct}`, bx, by + bh + 17);
    }
  }

  function update() {
    k = counts(prev, sens, fpr);
    const post = posterior(prev, sens, fpr);
    ro.set("sick", `1000명 중 환자 ${k.sick}명`);
    ro.set("nat", `양성 ${k.tp + k.fp}명 중 환자 ${k.tp}명`);
    ro.set("post", `P(병 | +) = ${post < 0.1 ? post.toFixed(4) : post.toFixed(3)}`, "hot");
    ro.set("sens", `P(+ | 병) = ${sens.toFixed(2)}`);
    pageState.set("baseRate", { prev, sens, fpr }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }
  let raf = 0;

  const pre = presets<PK>(PREVS.map((p) => ({ key: p.key, label: narrow ? p.label : `유병률 ${p.label}` })), (key) => {
    prev = PREVS.find((p) => p.key === key)!.v; pp.set(Math.log10(prev)); update();
  }, "p001");
  const pp = param({
    label: "유병률", min: -3, max: Math.log10(0.5), step: 0.01, value: Math.log10(prev),
    fmt: (v) => { const p = 10 ** v; return p < 0.01 ? `${(p * 100).toFixed(2)}%` : `${(p * 100).toFixed(1)}%`; },
    aria: "유병률(로그 눈금)", onInput: (v) => { prev = 10 ** v; pre.select(null); update(); },
  });
  const psn = param({ label: "민감도", min: 0.5, max: 1, step: 0.01, value: sens, aria: "민감도 P(+|병)", onInput: (v) => { sens = v; update(); } });
  const pf = param({ label: "위양성률", min: 0, max: 0.2, step: 0.005, value: fpr, fmt: (v) => v.toFixed(3), aria: "위양성률 P(+|정상)", onInput: (v) => { fpr = v; update(); } });
  // 이름이 세 글자라 이름 칸을 넓힌다
  for (const r of [pp.row, psn.row, pf.row]) r.style.gridTemplateColumns = "4.2em minmax(0, 1fr) 3.6em";
  const grid = el("div", "param-grid");
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(220px, 1fr))";
  grid.append(pp.row, psn.row, pf.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
