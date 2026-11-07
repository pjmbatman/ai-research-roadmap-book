// roofline 모델 (data/widgets.toml: roofline).
// 5.5장의 식 P_attainable = min(P_peak, I · B)와 같은 가상 장치(10 TFLOP/s, 1 TB/s, 지붕의 꺾이는 점 I* = 10 FLOP/byte)를 쓴다.
// 커널의 연산 강도 I = FLOP / byte. 메모리 이동량은 본문처럼 '최소 이동'(입력·출력·가중치를 한 번씩 읽고 쓴다)으로 센다.
//   벡터 덧셈 y = 2a + b (길이 n): 2n FLOP, 3n 원소
//   행렬곱 C = AB (m × m): 2m³ FLOP, 3m² 원소
//   합성곱 3×3 (64 → 64채널, 56 × 56, same padding): 2·Cout·Cin·k²·H·W FLOP, Cin·H·W + Cout·Cin·k² + Cout·H·W 원소
// dtype을 줄이면 원소당 byte가 줄어 I가 커진다(가상 장치의 최대 연산 속도는 그대로 둔다).
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

export const PEAK = 10e12, BW = 1e12, RIDGE = PEAK / BW;
export type Dtype = "f64" | "f32" | "f16";
export const BYTES: Record<Dtype, number> = { f64: 8, f32: 4, f16: 2 };
/** 커널: FLOP와 옮기는 원소 수 */
export const KERNELS: Record<string, { label: string; short: string; flops: number; elems: number }> = (() => {
  const n = 1e7, m = 4096, s = 64, cin = 64, cout = 64, k = 3, hw = 56 * 56;
  return {
    axpy: { label: "벡터 덧셈", short: "y=2a+b", flops: 2 * n, elems: 3 * n },
    mmSmall: { label: "행렬곱 m=64", short: "matmul 64", flops: 2 * s ** 3, elems: 3 * s * s },
    conv: { label: "합성곱 3×3", short: "conv 3×3", flops: 2 * cout * cin * k * k * hw, elems: cin * hw + cout * cin * k * k + cout * hw },
    mm: { label: "행렬곱 m=4096", short: "matmul 4096", flops: 2 * m ** 3, elems: 3 * m * m },
  };
})();
export const intensity = (key: string, dt: Dtype) => KERNELS[key].flops / (KERNELS[key].elems * BYTES[dt]);
export const attainable = (I: number) => Math.min(PEAK, I * BW);

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 560;
  const f = frame(root, "roofline 모델", "manipulate", narrow ? 0.85 : 0.5, "로그 축 위의 roofline: 연산 강도에 따른 최대 연산 속도와 커널들의 위치");
  let dtype: Dtype = "f64", kernel: string | null = "axpy";
  let logI = Math.log10(intensity("axpy", dtype));
  const LX0 = -2, LX1 = 3.5, LY0 = -3, LY1 = Math.log10(30); // I: 0.01–3000 FLOP/byte, P: 0.001–30 TFLOP/s

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const padL = 46, padR = 14, top = 30, bot = 40;
    const X = (lx: number) => padL + ((lx - LX0) / (LX1 - LX0)) * (W - padL - padR);
    const Y = (ly: number) => top + ((LY1 - ly) / (LY1 - LY0)) * (H - top - bot);
    const lp = (I: number) => Math.log10(attainable(I) / 1e12);
    const I = 10 ** logI, memBound = I < RIDGE;
    // 메모리 한계 영역
    ctx.fillStyle = alpha(C.ink2, 0.07);
    ctx.fillRect(X(LX0), Y(LY1), X(Math.log10(RIDGE)) - X(LX0), Y(LY0) - Y(LY1));
    // 격자와 축
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = C.mid; ctx.textBaseline = "middle";
    for (let e = LX0; e <= LX1; e++) {
      ctx.beginPath(); ctx.moveTo(X(e), Y(LY0)); ctx.lineTo(X(e), Y(LY1)); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText(e < 0 ? (10 ** e).toFixed(-e) : String(10 ** e), X(e), Y(LY0) + 12);
    }
    for (let e = LY0; e <= 1; e++) {
      ctx.beginPath(); ctx.moveTo(X(LX0), Y(e)); ctx.lineTo(X(LX1), Y(e)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(e < 0 ? (10 ** e).toFixed(-e) : String(10 ** e), padL - 6, Y(e));
    }
    ctx.textAlign = "center"; ctx.font = "600 11px Pretendard Variable, sans-serif";
    ctx.fillText("연산 강도 I (FLOP/byte, 로그)", (padL + W - padR) / 2, H - 10);
    ctx.textAlign = "left";
    ctx.fillText("최대 속도 (TFLOP/s, 로그)", 8, 13);
    ctx.textAlign = "right";
    ctx.fillText("가상 장치: 10 TFLOP/s · 1 TB/s", W - padR, 13);
    // 지붕
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(X(LX0), Y(lp(10 ** LX0)));
    ctx.lineTo(X(Math.log10(RIDGE)), Y(lp(RIDGE)));
    ctx.lineTo(X(LX1), Y(lp(10 ** LX1)));
    ctx.stroke();
    ctx.setLineDash([3, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(1), Y(LY0)); ctx.lineTo(X(1), Y(1)); ctx.stroke(); ctx.setLineDash([]);
    ctx.font = "600 11px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid; ctx.textAlign = "left";
    ctx.fillText("I* = 10", X(1) + 5, Y(LY0) - 10);
    // 커널들(흐리게)과 지금 고른 점. 지붕(연산 한계) 위의 점 이름은 겹치지 않게 아래로 층을 나눠 쓴다.
    ctx.font = "10px JetBrains Mono, monospace";
    let onRoof = 0;
    for (const [key, kk] of Object.entries(KERNELS)) {
      if (key === kernel) continue;
      const li = Math.log10(intensity(key, dtype));
      if (li < LX0 || li > LX1) continue;
      const x = X(li), y = Y(lp(10 ** li)), roof = 10 ** li >= RIDGE;
      ctx.fillStyle = C.mid;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill();
      const right = x + 7 + ctx.measureText(kk.short).width > W - padR;
      ctx.textAlign = right ? "right" : "left";
      ctx.fillText(kk.short, x + (right ? -7 : 7), y + (roof ? 14 + 13 * onRoof++ : 12));
    }
    const px = X(logI), py = Y(lp(I));
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.edge; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(px, Y(LY0)); ctx.lineTo(px, py); ctx.lineTo(X(LX0), py); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.edge;
    ctx.beginPath(); ctx.arc(px, py, 7, 0, 7); ctx.fill();
    ctx.font = "700 12px Pretendard Variable, sans-serif";
    const tag = `${kernel ? KERNELS[kernel].short + " · " : ""}${memBound ? "메모리 한계" : "연산 한계"}`;
    if (memBound) { ctx.textAlign = "left"; ctx.fillText(tag, px + 11, py + 16); }
    else { ctx.textAlign = logI > 1.6 ? "right" : "left"; ctx.fillText(tag, px + (logI > 1.6 ? -11 : 11), py - 14); }
    ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
  }

  const kBtns = presets(Object.entries(KERNELS).map(([k, v]) => ({ key: k, label: v.label })), (k) => { kernel = k; logI = Math.max(LX0, Math.min(LX1, Math.log10(intensity(k, dtype)))); pI.set(logI); update(); }, kernel ?? undefined);
  kBtns.row.setAttribute("aria-label", "커널");
  const dBtns = presets<Dtype>([{ key: "f64", label: "float64" }, { key: "f32", label: "float32" }, { key: "f16", label: "float16" }], (d) => {
    logI += Math.log10(BYTES[dtype] / BYTES[d]); // 같은 FLOP을 더 적은 byte로
    dtype = d;
    logI = Math.max(LX0, Math.min(LX1, logI));
    pI.set(logI); update();
  }, dtype);
  dBtns.row.setAttribute("aria-label", "dtype");
  const fmtI = (v: number) => { const x = 10 ** v; return x < 1 ? x.toFixed(3) : x < 100 ? x.toFixed(2) : x.toFixed(0); };
  const pI = param({ label: "I", min: LX0, max: LX1, step: 0.01, value: logI, fmt: fmtI, aria: "연산 강도 I (로그 눈금)", onInput: (v) => { logI = v; kernel = null; kBtns.select(null); update(); } });
  const grid = el("div", "param-grid");
  grid.append(pI.row);
  const ro = readouts();
  f.panel.append(kBtns.row, dBtns.row, grid, ro.row);

  function update() {
    const I = 10 ** logI, P = attainable(I);
    ro.set("I", `I = ${fmtI(logI)} FLOP/byte`);
    ro.set("P", `상한 ${(P / 1e12).toPrecision(3)} TFLOP/s · peak의 ${((P / PEAK) * 100).toPrecision(3)}%`);
    ro.set("bound", I < RIDGE ? "메모리 한계 (I < I*)" : "연산 한계 (I ≥ I*)", I < RIDGE ? "hot" : "ok");
    draw();
  }
  f.onResize(draw);
  update();
}
