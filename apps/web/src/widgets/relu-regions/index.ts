// ReLU 신경망의 조각별 아핀 영역 (data/widgets.toml: relu-regions).
// 입력 평면 [-3, 3]²의 각 점에서 은닉 뉴런의 켜짐·꺼짐(활성 패턴)을 계산해, 같은 패턴을 같은 색으로 칠한다.
// 같은 패턴의 입력에는 같은 아핀 사상이 적용된다. 영역 수는 화면 격자에서 센 값이므로 아주 작은 영역은 놓칠 수 있다(하한).
// 초기화는 labs/linear-algebra/affine-and-nonlinear/plane_warp.py와 같은 규칙(W ~ N(0, 1/n_in), b ~ N(0, 0.25))이다.
import { frame, param, presets, readouts, el, C, rng, rgbOf } from "../frame";
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
    // 영역 색: chart(blue) 계열 색조 안에서만 바꾸고, 바탕(--card)의 밝기에 맞춰 명도를 고른다
    const [br, bg2, bb2] = rgbOf(C.bg);
    const dark = br + bg2 + bb2 < 384;
    const hue = (205 + g * 60) / 360, sat = dark ? 0.35 : 0.55, light = (dark ? 16 + ((id * 37) % 20) : 78 + ((id * 37) % 17)) / 100;
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
      let c = id === hid ? rgbOf(C.edge) : cache.get(id);
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
      ctx.strokeStyle = C.mid;
      ctx.lineWidth = 1.2;
      const toPx = (x: number, y: number) => [ox + ((x + 3) / 6) * side, oy + ((3 - y) / 6) * side] as const;
      for (let r = 0; r < layers[0].W.length; r++) {
        const [w1, w2] = layers[0].W[r], bb = layers[0].b[r];
        const newest = r === layers[0].W.length - 1;   // 가장 최근에 더한 뉴런의 경계는 코랄
        ctx.strokeStyle = newest ? C.edge : C.mid;
        ctx.lineWidth = newest ? 2 : 1.2;
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
    ro.set("bound", depth === 1 ? `직선 ${n}개로 최대 ${maxLines}개` : `${depth}층 × ${width} 뉴런`);
    pageState.set("weights", layers.map((L) => ({ W: L.W, b: L.b })), root);
    draw();
  }

  const pw = param({ label: "폭", min: 1, max: 16, step: 1, value: width, fmt: (v) => `${v}`, aria: "층당 뉴런 수", onInput: (v) => { width = v; update(); } });
  // 단계형 그림 카드(benchmarks.md §3.4-18): 한 뉴런 → 두 뉴런 → 한 층 → 깊게
  const STEPS = {
    s1: { label: "1 뉴런", w: 1, d: 1, cap: "뉴런 하나는 평면에 직선 하나를 긋고, 그 양쪽에서 켜짐과 꺼짐이 갈린다." },
    s2: { label: "2 뉴런", w: 2, d: 1, cap: "직선 두 개가 평면을 최대 네 조각으로 나눈다. 조각마다 다른 아핀 사상이 적용된다." },
    s3: { label: "한 층", w: 6, d: 1, cap: "한 층의 직선 n개는 최대 1 + n + n(n−1)/2개 조각을 만든다. 폭을 바꿔 본다." },
    s4: { label: "깊게", w: 6, d: 2, cap: "층을 쌓으면 앞 층이 만든 조각을 다시 접어 나눈다. 경계가 꺾인 선이 된다." },
  } as const;
  type SK = keyof typeof STEPS;
  const cap = el("p", "wcap");
  const depthBtns = presets((Object.keys(STEPS) as SK[]).map((k) => ({ key: k, label: STEPS[k].label })), (k) => {
    width = STEPS[k].w; depth = STEPS[k].d; pw.set(width); cap.textContent = STEPS[k].cap; update();
  }, "s3");
  depthBtns.row.classList.add("segmented");
  cap.textContent = STEPS.s3.cap;
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
  f.panel.append(depthBtns.row, cap, grid, tools, ro.row);

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
