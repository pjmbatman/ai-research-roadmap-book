// ReLU 신경망의 조각별 아핀 영역 (data/widgets.toml: relu-regions).
// 입력 평면 [-3, 3]²의 각 점에서 은닉 뉴런의 켜짐·꺼짐(활성 패턴)을 계산해, 같은 패턴을 같은 색으로 칠한다.
// 같은 패턴의 입력에는 같은 아핀 사상이 적용된다. 영역 수는 화면 격자에서 센 값이므로 아주 작은 영역은 놓칠 수 있다(하한).
// 초기화는 labs/linear-algebra/affine-and-nonlinear/plane_warp.py와 같은 규칙(W ~ N(0, 1/n_in), b ~ N(0, 0.25))이다.
import { frame, param, presets, readouts, el, C, rng } from "../frame";
import { pageState } from "../../client/store";

type Layer = { W: number[][]; b: number[] };
const RES = 220;

function init(widths: number[], seed: number): Layer[] {
  const r = rng(seed * 7919 + 17);
  const layers: Layer[] = [];
  for (let i = 0; i + 1 < widths.length; i++) {
    const [n, m] = [widths[i], widths[i + 1]];
    layers.push({
      W: Array.from({ length: m }, () => Array.from({ length: n }, () => r.normal() / Math.sqrt(n))),
      b: Array.from({ length: m }, () => r.normal() * 0.5),
    });
  }
  return layers;
}

export function mount(root: HTMLElement) {
  const f = frame(root, "ReLU 신경망의 조각별 아핀 영역", "manipulate", 0.62, "활성 패턴별로 색칠한 입력 평면과 첫 층 뉴런의 경계선");
  let width = 6, depth = 1, seed = 1, showLines = true;
  let layers: Layer[] = [];
  let ids = new Int32Array(RES * RES);
  let count = 0;
  let hover: [number, number] | null = null;

  function compute() {
    layers = init([2, ...Array(depth).fill(width)], seed);
    const map = new Map<string, number>();
    const hidden = width * depth;
    const bits = new Uint8Array(hidden);
    for (let j = 0; j < RES; j++) {
      for (let i = 0; i < RES; i++) {
        let h = [-3 + (6 * (i + 0.5)) / RES, 3 - (6 * (j + 0.5)) / RES];
        let k = 0;
        for (const L of layers) {
          const z = L.W.map((row, r) => row.reduce((s, w, c) => s + w * h[c], L.b[r]));
          z.forEach((v) => (bits[k++] = v > 0 ? 1 : 0));
          h = z.map((v) => Math.max(0, v));
        }
        const key = String.fromCharCode(...bits);
        let id = map.get(key);
        if (id === undefined) { id = map.size; map.set(key, id); }
        ids[j * RES + i] = id;
      }
    }
    count = map.size;
  }

  const off = document.createElement("canvas");
  off.width = off.height = RES;
  const octx = off.getContext("2d")!;
  const img = octx.createImageData(RES, RES);
  const rgb = (id: number): [number, number, number] => {
    // Glacier 계열의 차가운 색조 안에서만 바꾼다(엣지 색은 강조에만 쓴다)
    const g = (id * 0.6180339887) % 1;
    const hue = (205 + g * 60) / 360, sat = 0.38, light = (14 + ((id * 37) % 23)) / 100;
    const q = light < 0.5 ? light * (1 + sat) : light + sat - light * sat, pp = 2 * light - q;
    const ch = (t: number) => { t = (t + 1) % 1; return t < 1 / 6 ? pp + (q - pp) * 6 * t : t < 0.5 ? q : t < 2 / 3 ? pp + (q - pp) * (2 / 3 - t) * 6 : pp; };
    return [ch(hue + 1 / 3) * 255, ch(hue) * 255, ch(hue - 1 / 3) * 255];
  };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const side = Math.min(W - 24, H - 24);
    const ox = (W - side) / 2, oy = (H - side) / 2;
    const hid = hover ? ids[hover[1] * RES + hover[0]] : -1;
    const cache = new Map<number, [number, number, number]>();
    for (let k = 0; k < RES * RES; k++) {
      const id = ids[k];
      let c = id === hid ? ([255, 107, 90] as [number, number, number]) : cache.get(id);
      if (!c) { c = rgb(id); cache.set(id, c); }
      img.data[4 * k] = c[0]; img.data[4 * k + 1] = c[1]; img.data[4 * k + 2] = c[2]; img.data[4 * k + 3] = id === hid ? 170 : 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, ox, oy, side, side);
    if (showLines) {
      // 첫 층 뉴런의 경계 w·x + b = 0 (직선). 깊은 층의 경계는 꺾인 선이라 색 경계로만 보인다.
      ctx.save();
      ctx.beginPath(); ctx.rect(ox, oy, side, side); ctx.clip();
      ctx.strokeStyle = "rgba(220,230,242,0.65)";
      ctx.lineWidth = 1.2;
      const toPx = (x: number, y: number) => [ox + ((x + 3) / 6) * side, oy + ((3 - y) / 6) * side] as const;
      for (let r = 0; r < layers[0].W.length; r++) {
        const [w1, w2] = layers[0].W[r], bb = layers[0].b[r];
        ctx.beginPath();
        if (Math.abs(w2) > Math.abs(w1)) { ctx.moveTo(...toPx(-4, (-bb + 4 * w1) / w2)); ctx.lineTo(...toPx(4, (-bb - 4 * w1) / w2)); }
        else { ctx.moveTo(...toPx((-bb + 4 * w2) / w1, -4)); ctx.lineTo(...toPx((-bb - 4 * w2) / w1, 4)); }
        ctx.stroke();
      }
      ctx.restore();
    }
    ctx.strokeStyle = C.faint;
    ctx.strokeRect(ox, oy, side, side);
  }

  function update() {
    compute();
    const n = width;
    const maxLines = 1 + n + (n * (n - 1)) / 2;
    ro.set("count", `활성 영역 ≥ ${count}개`, "hot");
    ro.set("neurons", `은닉 뉴런 ${width * depth}개 (${depth}층 × ${width})`);
    ro.set("bound", depth === 1 ? `직선 ${n}개의 최대 영역 수 ${maxLines}` : `첫 층만이면 최대 ${maxLines}`);
    pageState.set("weights", layers.map((L) => ({ W: L.W, b: L.b })), root);
    draw();
  }

  const pw = param({ label: "폭", min: 1, max: 16, step: 1, value: width, fmt: (v) => `${v}`, aria: "층당 뉴런 수", onInput: (v) => { width = v; update(); } });
  const depthBtns = presets([{ key: "1", label: "은닉 1층" }, { key: "2", label: "은닉 2층" }, { key: "3", label: "은닉 3층" }], (k) => { depth = Number(k); update(); }, "1");
  const tools = el("div", "wframe-row");
  const reseed = el("button", "preset");
  reseed.type = "button";
  reseed.textContent = "↻ 새 가중치";
  reseed.addEventListener("click", () => { seed++; update(); });
  const lines = el("label", "toggle");
  lines.innerHTML = `<input type="checkbox" checked /> 첫 층 경계선`;
  lines.querySelector("input")!.addEventListener("change", (e) => { showLines = (e.target as HTMLInputElement).checked; draw(); });
  tools.append(reseed, lines);
  const grid = el("div", "param-grid");
  grid.append(pw.row);
  const ro = readouts();
  const note = el("small", "", "같은 색 = 같은 활성 패턴 = 같은 아핀 사상. 영역에 마우스를 올리면 그 조각이 강조된다. 같은 뉴런 수에서 폭과 깊이를 바꿔 비교해 보자(예: 1층 × 12 vs 2층 × 6).");
  note.style.color = "var(--mute)";
  f.panel.append(depthBtns.row, grid, tools, ro.row, note);

  f.canvas.addEventListener("pointermove", (e) => {
    const r = f.canvas.getBoundingClientRect();
    const side = Math.min(f.width - 24, f.height - 24);
    const ox = (f.width - side) / 2, oy = (f.height - side) / 2;
    const i = Math.floor(((e.clientX - r.left - ox) / side) * RES), j = Math.floor(((e.clientY - r.top - oy) / side) * RES);
    const nh = i >= 0 && j >= 0 && i < RES && j < RES ? ([i, j] as [number, number]) : null;
    if (String(nh) !== String(hover)) { hover = nh; draw(); }
  });
  f.canvas.addEventListener("pointerleave", () => { hover = null; draw(); });
  f.onResize(draw);
  update();
}
