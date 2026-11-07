// 컬러맵 밝기 (data/widgets.toml: colormap-lightness).
// 23.9장의 계산: 컬러맵을 [0, 1]의 256점에서 sRGB로 읽고, 본문의 lightness()와 같은 식으로 CIELAB L*를 잰다.
// jet은 matplotlib의 구간 선형 정의(_segmentdata)를 그대로 쓰고, viridis는 matplotlib 표를 33점으로 줄여 선형 보간한다
// (256점 표와의 L* 차이 0.3 이하). 읽음값: 양 끝 L*, 최대 L* 위치, 단조 증가(본문과 같이 이웃 차이 > 0.01).
// 아래 그림은 가상의 2차원 장(완만한 기울기 + 봉우리 하나)이다. '흑백으로'는 각 색을 같은 L*의 회색으로 바꾼다.
import { frame, presets, readouts, el, C } from "../frame";

const VIRIDIS = ["440154", "470d60", "48186a", "482374", "472d7b", "453781", "424086", "3e4989", "3b528b", "375b8d", "33638d", "2f6b8e", "2c728e", "297a8e", "26828e", "23898e", "21918c", "1f988b", "1fa088", "22a785", "28ae80", "32b67a", "3fbc73", "4ec36b", "5ec962", "70cf57", "84d44b", "98d83e", "addc30", "c2df23", "d8e219", "ece51b", "fde725"]
  .map((h) => [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255));
type Seg = [number, number][];
const JET: Record<"r" | "g" | "b", Seg> = {
  r: [[0, 0], [0.35, 0], [0.66, 1], [0.89, 1], [1, 0.5]],
  g: [[0, 0], [0.125, 0], [0.375, 1], [0.64, 1], [0.91, 0], [1, 0]],
  b: [[0, 0.5], [0.11, 1], [0.34, 1], [0.65, 0], [1, 0]],
};
const seg = (s: Seg, x: number) => {
  for (let i = 1; i < s.length; i++) if (x <= s[i][0]) { const [x0, y0] = s[i - 1], [x1, y1] = s[i]; return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0); }
  return s[s.length - 1][1];
};
type CM = "viridis" | "jet" | "gray";
export function cmap(name: CM, x: number): [number, number, number] {
  x = Math.max(0, Math.min(1, x));
  if (name === "gray") return [x, x, x];
  if (name === "jet") return [seg(JET.r, x), seg(JET.g, x), seg(JET.b, x)];
  const t = x * (VIRIDIS.length - 1), i = Math.min(Math.floor(t), VIRIDIS.length - 2), u = t - i;
  return [0, 1, 2].map((k) => VIRIDIS[i][k] + (VIRIDIS[i + 1][k] - VIRIDIS[i][k]) * u) as [number, number, number];
}
/** 본문 lightness(): sRGB(0–1) → 상대 휘도 Y → CIELAB L* */
export function lightness([r, g, b]: [number, number, number]) {
  const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const Y = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return Y > (6 / 29) ** 3 ? 116 * Math.cbrt(Y) - 16 : Y * (29 / 3) ** 3;
}
/** L*에서 같은 밝기의 회색 sRGB 값(0–1)으로 */
function grayOf(L: number) {
  const Y = L > 8 ? ((L + 16) / 116) ** 3 : L / (29 / 3) ** 3;
  return Y <= 0.0031308 ? 12.92 * Y : 1.055 * Y ** (1 / 2.4) - 0.055;
}
export function profile(name: CM) {
  const xs = Array.from({ length: 256 }, (_, i) => i / 255);
  const L = xs.map((x) => lightness(cmap(name, x)));
  let imax = 0; L.forEach((v, i) => { if (v > L[imax]) imax = i; });
  const mono = L.every((v, i) => i === 0 || v - L[i - 1] > 0.01);
  return { xs, L, start: L[0], end: L[255], argmax: xs[imax], mono };
}
// 가상의 2차원 장: 왼쪽에서 오른쪽으로 완만히 오르는 기울기 + 봉우리 하나, [0, 1]로 정규화
const field = (x: number, y: number) => 0.55 * x + 0.45 * Math.exp(-((x - 0.62) ** 2 + (y - 0.5) ** 2) / 0.05);

const NAMES: Record<CM, string> = { viridis: "viridis", jet: "jet", gray: "gray" };

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "컬러맵 밝기", "compare", narrow ? 1.2 : 0.52, "컬러맵의 밝기 L* 곡선과, 같은 가상 2차원 장을 그 컬러맵으로 칠한 그림");
  let name: CM = (attrs.cmap as CM) in NAMES ? (attrs.cmap as CM) : "jet";
  let mono = false;
  const RES = 120;
  const off = document.createElement("canvas");
  off.width = off.height = RES;
  const octx = off.getContext("2d")!;
  const img = octx.createImageData(RES, RES);
  let fmin = Infinity, fmax = -Infinity;
  const vals = new Float32Array(RES * RES);
  for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) {
    const v = field((i + 0.5) / RES, 1 - (j + 0.5) / RES);
    vals[j * RES + i] = v; fmin = Math.min(fmin, v); fmax = Math.max(fmax, v);
  }
  const shown = (x: number): [number, number, number] => {
    const c = cmap(name, x);
    if (!mono) return c;
    const g = grayOf(lightness(c));
    return [g, g, g];
  };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const p = profile(name);
    // 왼쪽(위): L* 곡선과 색 띠
    const lw = narrow ? W : W * 0.55, lh = narrow ? H * 0.5 : H;
    const axL = 40, pt = 28, stripH = 14, pb = lh - 40;
    const X = (x: number) => axL + x * (lw - axL - 12);
    const Y = (L: number) => pb - (L / 100) * (pb - pt);
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.ink;
    ctx.fillText(`${name}의 밝기 L*`, 6, 16);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    for (const L of [0, 25, 50, 75, 100]) {
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(axL, Y(L)); ctx.lineTo(lw - 12, Y(L)); ctx.stroke();
      const s = String(L);
      ctx.fillText(s, axL - 5 - ctx.measureText(s).width, Y(L) + 4);
    }
    // 색 띠(그림에 쓰는 색 그대로)
    const sw = lw - axL - 12;
    for (let i = 0; i < 128; i++) {
      const [r, g, b] = shown((i + 0.5) / 128);
      ctx.fillStyle = `rgb(${r * 255},${g * 255},${b * 255})`;
      ctx.fillRect(axL + (sw * i) / 128, pb + 6, sw / 128 + 0.6, stripH);
    }
    ctx.fillStyle = C.mid;
    for (const x of [0, 0.5, 1]) { const s = x.toFixed(1); ctx.fillText(s, Math.min(X(x) - ctx.measureText(s).width / 2, lw - 12 - ctx.measureText(s).width), pb + stripH + 18); }
    // 곡선: 단조가 아니면 감소 구간을 코랄로
    ctx.lineWidth = 2.4;
    for (let i = 1; i < 256; i++) {
      ctx.strokeStyle = p.L[i] - p.L[i - 1] > 0.01 ? C.ink2 : C.edge;
      ctx.beginPath(); ctx.moveTo(X(p.xs[i - 1]), Y(p.L[i - 1])); ctx.lineTo(X(p.xs[i]), Y(p.L[i])); ctx.stroke();
    }
    ctx.fillStyle = C.edge;
    ctx.beginPath(); ctx.arc(X(p.argmax), Y(Math.max(...p.L)), 4.5, 0, 7); ctx.fill();
    // 오른쪽(아래): 같은 데이터
    const gx0 = narrow ? 0 : W * 0.58, gy0 = narrow ? H * 0.52 : 0, gw = narrow ? W : W * 0.42, gh = narrow ? H * 0.48 : H;
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.ink;
    ctx.fillText(mono ? "가상 2차원 장 · 흑백" : "가상 2차원 장", gx0 + 6, gy0 + 16);
    const side = Math.min(gw - 16, gh - 32);
    const ox = gx0 + (gw - side) / 2, oy = gy0 + 24 + (gh - 24 - side) / 2;
    const cache = new Map<number, [number, number, number]>();
    for (let k = 0; k < RES * RES; k++) {
      const q = Math.round(((vals[k] - fmin) / (fmax - fmin)) * 255);
      let c = cache.get(q);
      if (!c) { c = shown(q / 255); cache.set(q, c); }
      img.data[4 * k] = c[0] * 255; img.data[4 * k + 1] = c[1] * 255; img.data[4 * k + 2] = c[2] * 255; img.data[4 * k + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, ox, oy, side, side);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, side, side);
  }

  function update() {
    const p = profile(name);
    ro.set("ends", `L* ${p.start.toFixed(1)} → ${p.end.toFixed(1)}`, Math.abs(p.end - p.start) < 20 ? "hot" : "");
    ro.set("max", `최대 L* 위치 ${p.argmax.toFixed(2)}`, p.argmax < 0.99 ? "hot" : "");
    ro.set("mono", p.mono ? "단조 증가: 예" : "단조 증가: 아니오", p.mono ? "ok" : "hot");
    draw();
  }

  const btns = presets<CM>([{ key: "viridis", label: "viridis" }, { key: "jet", label: "jet" }, { key: "gray", label: "gray" }], (k) => { name = k; update(); }, name);
  btns.row.classList.add("segmented");
  const tools = el("div", "wframe-row");
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" /> 흑백으로`;
  sw.querySelector("input")!.addEventListener("change", (e) => { mono = (e.target as HTMLInputElement).checked; update(); });
  tools.append(sw);
  const ro = readouts();
  f.panel.append(btns.row, tools, ro.row);
  f.onResize(draw);
  update();
}
