// 축 범위와 lie factor (data/widgets.toml: axis-lie-factor).
// 23.9장의 예: 정확도 A 0.912, B 0.925를 막대로 그리고 y축 아래 끝 y0를 바꾼다.
// 본문의 정의: 그림에서 보이는 변화율 ((b − y0) − (a − y0)) / (a − y0)을 데이터의 변화율 (b − a) / a로 나눈 값.
// 정리하면 a / (a − y0)이고, y0 = 0.905에서 130.3, y0 = 0에서 1이다. 점으로 그리면 값을 위치로 읽으므로 길이 비교가 사라진다.
import { frame, param, presets, readouts, el, C } from "../frame";

const A = 0.912, B = 0.925;
export function lieFactor(y0: number, a = A, b = B) {
  const shown = ((b - y0) - (a - y0)) / (a - y0);
  return shown / ((b - a) / a);
}
/** 축 위 끝: B 막대가 축 높이의 약 86%가 되게 하되 1을 넘지 않는다(y0 = 0이면 1, y0 = 0.905이면 약 0.928). */
export const yTop = (y0: number) => Math.min(1, y0 + (B - y0) / 0.86);

const PRESETS = {
  zero: { label: "0부터", y0: 0 },
  half: { label: "0.5부터", y0: 0.5 },
  cut: { label: "본문 자름 0.905", y0: 0.905 },
} as const;
type PK = keyof typeof PRESETS;

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "축 범위와 lie factor", "manipulate", narrow ? 1.15 : 0.5, "정확도 두 값을 그린 막대 그림과 축 아래 끝에 따른 막대 높이 비");
  const start: PK = (attrs.preset as PK) in PRESETS ? (attrs.preset as PK) : "cut";
  let y0: number = PRESETS[start].y0;
  let dots = false;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const top = yTop(y0);
    // 왼쪽: 막대(또는 점) 그림, 오른쪽: lie factor 곡선
    const lw = narrow ? W : W * 0.52;
    const ph = narrow ? H * 0.56 : H;
    const axL = 56, pt = 30, pb = ph - 30;
    const Y = (v: number) => pb - ((v - y0) / (top - y0)) * (pb - pt);
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.ink;
    ctx.fillText(dots ? "정확도(점: 위치로 읽음)" : "정확도(막대: 길이로 읽음)", 6, 16);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(axL, pt - 4); ctx.lineTo(axL, pb); ctx.lineTo(lw - 10, pb); ctx.stroke();
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    // 눈금: 1·2·5 × 10^k 간격, 아래 끝 y0는 따로 적는다(눈금과 너무 가까우면 생략)
    const raw = (top - y0) / 5, mag = 10 ** Math.floor(Math.log10(raw));
    const stp = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw)!;
    const digits = Math.max(0, -Math.floor(Math.log10(stp)));
    const tick = (v: number, s: string) => {
      ctx.fillText(s, axL - 5 - ctx.measureText(s).width, Y(v) + 4);
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(axL - 3, Y(v)); ctx.lineTo(axL, Y(v)); ctx.stroke();
    };
    for (let v = Math.ceil(y0 / stp - 1e-9) * stp; v <= top + 1e-9; v += stp) if (v - y0 > stp * 0.45) tick(v, v.toFixed(digits));
    tick(y0, y0.toFixed(Math.max(digits, y0 === 0 ? 0 : 3)));
    const xs = [axL + (lw - axL) * 0.3, axL + (lw - axL) * 0.72];
    const bw = Math.min(56, (lw - axL) * 0.24);
    [[A, "A", C.ink2], [B, "B", C.edge]].forEach(([v, name, col], i) => {
      const val = v as number;
      ctx.fillStyle = col as string;
      if (dots) {
        ctx.strokeStyle = C.grid2; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(axL, Y(val)); ctx.lineTo(xs[i], Y(val)); ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(xs[i], Y(val), 6, 0, 7); ctx.fill();
      } else {
        ctx.fillRect(xs[i] - bw / 2, Y(val), bw, pb - Y(val));
      }
      ctx.fillStyle = C.ink; ctx.font = "11px JetBrains Mono, monospace";
      const s = val.toFixed(3);
      ctx.fillText(s, xs[i] - ctx.measureText(s).width / 2, Y(val) - 8);
      ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
      ctx.fillText(name as string, xs[i] - 4, pb + 18);
    });
    // 오른쪽(또는 아래): y0에 따른 lie factor (로그 축)
    const gx0 = narrow ? 0 : W * 0.56, gy0 = narrow ? H * 0.6 : 0, gw = narrow ? W : W * 0.44, gh = narrow ? H * 0.4 : H;
    const gl = gx0 + 44, gr = gx0 + gw - 12, gt = gy0 + 30, gb = gy0 + gh - 28;
    const Ymax = 3; // 10^3
    const GX = (v: number) => gl + (v / 0.91) * (gr - gl);
    const GY = (lf: number) => gb - (Math.log10(Math.max(1, lf)) / Ymax) * (gb - gt);
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.ink;
    ctx.fillText("y0에 따른 lie factor (로그 축)", gx0 + 6, gy0 + 16);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(gl, gt - 4); ctx.lineTo(gl, gb); ctx.lineTo(gr, gb); ctx.stroke();
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    for (let p = 0; p <= Ymax; p++) {
      const s = p === 0 ? "1" : `10${["", "¹", "²", "³"][p]}`;
      ctx.fillText(s, gl - 5 - ctx.measureText(s).width, GY(10 ** p) + 4);
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(gl, GY(10 ** p)); ctx.lineTo(gr, GY(10 ** p)); ctx.stroke();
    }
    for (const v of [0, 0.3, 0.6, 0.9]) {
      const s = v.toFixed(1), tw = ctx.measureText(s).width;
      ctx.fillText(s, Math.min(GX(v) - tw / 2, gr - tw), gb + 14);
    }
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const v = (0.91 * i) / 200;
      const p = [GX(v), GY(lieFactor(v))] as const;
      if (i) ctx.lineTo(...p); else ctx.moveTo(...p);
    }
    ctx.stroke();
    if (!dots) {
      ctx.fillStyle = C.edge;
      ctx.beginPath(); ctx.arc(GX(y0), GY(lieFactor(y0)), 5, 0, 7); ctx.fill();
    }
  }

  function update() {
    const ratioShown = (B - y0) / (A - y0), ratioData = B / A;
    const lf = lieFactor(y0);
    ro.set("h", dots ? "점: 높이 비교 없음" : `막대 높이 비 B/A ${ratioShown.toFixed(2)}`);
    ro.set("v", `값의 비 B/A ${ratioData.toFixed(3)} (+${(((B - A) / A) * 100).toFixed(1)}%)`);
    ro.set("lf", dots ? "lie factor: 길이를 읽지 않음" : `lie factor ${lf.toFixed(1)}`, dots ? "ok" : lf > 1.05 ? "hot" : "ok");
    draw();
  }

  const py = param({ label: "y0", min: 0, max: 0.91, step: 0.001, value: y0, fmt: (v) => v.toFixed(3), aria: "y축의 아래 끝 값", onInput: (v) => { y0 = v; pre.select(null); update(); } });
  const pre = presets((Object.keys(PRESETS) as PK[]).map((k) => ({ key: k, label: PRESETS[k].label })), (k) => { y0 = PRESETS[k].y0; py.set(y0); update(); }, start);
  const grid = el("div", "param-grid");
  grid.append(py.row);
  const tools = el("div", "wframe-row");
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" /> 막대 대신 점`;
  sw.querySelector("input")!.addEventListener("change", (e) => { dots = (e.target as HTMLInputElement).checked; update(); });
  tools.append(sw);
  const ro = readouts();
  f.panel.append(pre.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
