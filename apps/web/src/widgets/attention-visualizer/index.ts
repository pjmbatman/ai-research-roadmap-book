// Attention 계산 (data/widgets.toml: attention-visualizer).
// 토큰 6개(마지막은 PAD)의 Q·K(d_k = 16)·V(d_v = 4)로 scaled dot-product attention을 계산한다.
//   점수 S = Q Kᵀ / τ → 마스크(가려진 칸 = −∞) → 가중치 A = softmax(S) (행마다) → 출력 O = A V
// τ = √d_k가 표준 스케일이다. τ를 줄이면 softmax가 한 키에 몰리고(엔트로피↓), 키우면 고르게 퍼진다.
// 고른 쿼리 행에서 V의 각 행이 가중치만큼 섞여 출력 O_i가 되는 것을 오른쪽에 보인다.
import { frame, param, readouts, el, C, rng, rgbOf } from "../frame";
import { pageState } from "../../client/store";

const TOK = ["오늘", "나는", "새", "책을", "샀다", "PAD"];
const NT = TOK.length, DK = 16, DV = 4, PAD = NT - 1;
const SQRT_D = Math.sqrt(DK);

const r0 = rng(4250);
const mat = (rows: number, cols: number) => Array.from({ length: rows }, () => Array.from({ length: cols }, () => r0.normal()));
const Q = mat(NT, DK), K = mat(NT, DK), V = mat(NT, DV);
const lab = 38, gap = 16, head = 42, pad = 10, wlab = 46;
const RAW = Q.map((q) => K.map((k) => q.reduce((s, v, c) => s + v * k[c], 0))); // Q Kᵀ

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 460;
  const f = frame(root, "Attention 계산", "manipulate", narrow ? 1.2 : 0.56, "점수 행렬, softmax 가중치 행렬, 값 V와 고른 쿼리의 출력");
  let logTau = 2, query = 3, causal = false, padMask = true;
  let S: number[][] = [], A: number[][] = [], O: number[][] = [];
  const masked = (i: number, j: number) => (causal && j > i) || (padMask && j === PAD);

  function compute() {
    const tau = 2 ** logTau;
    S = RAW.map((row) => row.map((v) => v / tau));
    A = S.map((row, i) => {
      let m = -Infinity;
      row.forEach((v, j) => { if (!masked(i, j)) m = Math.max(m, v); });
      const e = row.map((v, j) => (masked(i, j) ? 0 : Math.exp(v - m)));
      const z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / z);
    });
    O = A.map((a) => Array.from({ length: DV }, (_, c) => a.reduce((s, w, j) => s + w * V[j][c], 0)));
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const { stacked, c } = layout(W, H);
    const x0 = pad + lab;
    const pS = { x: x0, y: head };
    const pA = { x: x0 + NT * c + gap, y: head };
    const pV = stacked ? { x: x0, y: head * 2 + NT * c + pad } : { x: pA.x + NT * c + gap, y: head };
    const smax = Math.max(...RAW.flat().map(Math.abs)) / 2 ** logTau || 1;
    const vmax = Math.max(...V.flat().map(Math.abs));
    const [pe, pm] = [rgbOf(C.edge), rgbOf(C.mid)];
    const rgba = (p: number[], a: number) => `rgba(${p[0]},${p[1]},${p[2]},${Math.max(0, Math.min(1, a)).toFixed(3)})`;
    const ink = (pos: boolean, a: number) => rgba(pos ? pe : pm, a);
    const nums = c >= 26;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";

    const cell = (x: number, y: number, fill: string) => { ctx.fillStyle = fill; ctx.fillRect(x + 1, y + 1, c - 2, c - 2); };
    const hatch = (x: number, y: number) => {
      ctx.save(); ctx.beginPath(); ctx.rect(x + 1, y + 1, c - 2, c - 2); ctx.clip();
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
      for (let k = -c; k < c; k += 5) { ctx.beginPath(); ctx.moveTo(x + k, y + c); ctx.lineTo(x + k + c, y); ctx.stroke(); }
      ctx.restore();
    };
    const title = (t: string, x: number, y: number) => {
      ctx.textAlign = "left"; ctx.fillStyle = C.mid; ctx.font = "600 12px Pretendard Variable, sans-serif";
      ctx.fillText(t, x, y - 26);
      ctx.textAlign = "center";
    };
    const colLabels = (x: number, y: number) => {
      ctx.font = `${c >= 26 ? 11 : 9}px Pretendard Variable, sans-serif`;
      TOK.forEach((t, j) => { ctx.fillStyle = j === PAD && padMask ? C.faint : C.mid; ctx.fillText(t, x + j * c + c / 2, y - 8); });
    };
    const rowLabels = (y: number) => {
      ctx.textAlign = "right"; ctx.font = "11px Pretendard Variable, sans-serif";
      TOK.forEach((t, i) => { ctx.fillStyle = i === query ? C.ink : C.mid; ctx.fillText(t, x0 - 6, y + i * c + c / 2); });
      ctx.textAlign = "center";
    };
    const rowBox = (x: number, y: number, w: number) => { ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.strokeRect(x + 0.5, y + query * c + 0.5, w - 1, c - 1); };

    // 점수
    title("점수 QKᵀ/τ", pS.x, pS.y);
    colLabels(pS.x, pS.y);
    rowLabels(pS.y);
    for (let i = 0; i < NT; i++) for (let j = 0; j < NT; j++) {
      const x = pS.x + j * c, y = pS.y + i * c;
      if (masked(i, j)) { hatch(x, y); continue; }
      const v = S[i][j];
      cell(x, y, ink(v >= 0, Math.min(1, Math.abs(v) / smax) * 0.85 + 0.05));
      if (nums) { ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = Math.abs(v) / smax > 0.55 ? C.bg : C.ink; ctx.fillText((Math.abs(v) < 0.05 ? 0 : v).toFixed(1), x + c / 2, y + c / 2 + 1); }
    }
    rowBox(pS.x, pS.y, NT * c);

    // 가중치
    title("가중치 softmax", pA.x, pA.y);
    colLabels(pA.x, pA.y);
    for (let i = 0; i < NT; i++) for (let j = 0; j < NT; j++) {
      const x = pA.x + j * c, y = pA.y + i * c;
      if (masked(i, j)) { hatch(x, y); continue; }
      const a = A[i][j];
      cell(x, y, rgba(pe, 0.04 + 0.96 * a));
      if (nums) { ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = a > 0.5 ? C.bg : C.ink; ctx.fillText(a.toFixed(2).replace(/^0/, ""), x + c / 2, y + c / 2 + 1); }
    }
    rowBox(pA.x, pA.y, NT * c);

    // V와 출력 O_i: V의 각 행을 가중치 A_ij만큼 진하게, 그 합이 O_i
    title(`V → 출력 O(${TOK[query]})`, pV.x, pV.y);
    if (stacked) rowLabels(pV.y);
    for (let j = 0; j < NT; j++) {
      const w = A[query][j];
      for (let k = 0; k < DV; k++) {
        const v = V[j][k];
        cell(pV.x + k * c, pV.y + j * c, ink(v >= 0, (Math.abs(v) / vmax) * (0.12 + 0.88 * w)));
      }
      ctx.fillStyle = C.mid; ctx.font = "10px JetBrains Mono, monospace"; ctx.textAlign = "left";
      if (pV.x + DV * c + wlab <= W) ctx.fillText(masked(query, j) ? "—" : `×${w.toFixed(2)}`, pV.x + DV * c + 4, pV.y + j * c + c / 2 + 1);
      ctx.textAlign = "center";
    }
    const oy = pV.y + NT * c + Math.round(c * 0.5);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pV.x, oy - c * 0.25); ctx.lineTo(pV.x + DV * c, oy - c * 0.25); ctx.stroke();
    for (let k = 0; k < DV; k++) {
      const v = O[query][k];
      cell(pV.x + k * c, oy, ink(v >= 0, Math.min(1, Math.abs(v) / vmax) * 0.9 + 0.05));
      if (nums) { ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = Math.abs(v) / vmax > 0.55 ? C.bg : C.ink; ctx.fillText((Math.abs(v) < 0.05 ? 0 : v).toFixed(1), pV.x + k * c + c / 2, oy + c / 2 + 1); }
    }
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.strokeRect(pV.x + 0.5, oy + 0.5, DV * c - 1, c - 1);
    ctx.textAlign = "left"; ctx.fillStyle = C.mid; ctx.font = "11px Pretendard Variable, sans-serif";
    ctx.textAlign = "right"; ctx.fillText("O", pV.x - 6, oy + c / 2 + 1);
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  }

  function update() {
    compute();
    const tau = 2 ** logTau;
    const row = A[query];
    let best = 0; row.forEach((v, j) => { if (v > row[best]) best = j; });
    const vis = row.filter((_, j) => !masked(query, j)).length;
    const H = -row.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0);
    const vals: number[] = [];
    S.forEach((rw, i) => rw.forEach((v, j) => { if (!masked(i, j)) vals.push(v); }));
    const mu = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mu) ** 2, 0) / vals.length);
    ro.set("max", `최대 가중치 ${row[best].toFixed(2)} → ${TOK[best]}`, "hot");
    ro.set("ent", `엔트로피 ${H.toFixed(2)} / ln ${vis} = ${Math.log(vis).toFixed(2)}`);
    ro.set("sd", `점수 표준편차 ${sd.toFixed(2)}`);
    ro.set("tau", `τ/√d ${(tau / SQRT_D).toFixed(2)}`, Math.abs(tau - SQRT_D) < 1e-9 ? "ok" : "");
    pageState.set("attention", { tau, query: TOK[query], causal, padMask, weights: A }, root);
    draw();
  }

  const pq = param({ label: "쿼리", min: 0, max: NT - 1, step: 1, value: query, fmt: (v) => TOK[v], aria: "쿼리 토큰", onInput: (v) => { query = v; update(); } });
  const pt = param({ label: "τ", min: -2, max: 4, step: 0.25, value: logTau, fmt: (v) => (2 ** v).toFixed(2), aria: "점수를 나누는 온도 τ (표준값은 √d = 4)", onInput: (v) => { logTau = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pq.row, pt.row);
  const tools = el("div", "wframe-row");
  const sw = (label: string, on: boolean, fn: (v: boolean) => void) => {
    const t = el("label", "toggle");
    t.innerHTML = `<input type="checkbox" ${on ? "checked" : ""} /> `;
    t.append(label);
    t.querySelector("input")!.addEventListener("change", (e) => { fn((e.target as HTMLInputElement).checked); update(); });
    return t;
  };
  tools.append(sw("causal mask", causal, (v) => (causal = v)), sw("padding mask", padMask, (v) => (padMask = v)));
  const ro = readouts();
  f.panel.append(grid, tools, ro.row);

  // 그림에서 행(쿼리)을 눌러 고른다
  f.canvas.addEventListener("pointerdown", (e) => {
    const r = f.canvas.getBoundingClientRect();
    const y = e.clientY - r.top;
    const { c } = layout(f.width, f.height);
    const i = Math.floor((y - head) / c);
    if (i >= 0 && i < NT) { query = i; pq.set(i); update(); }
  });
  f.canvas.style.cursor = "pointer";
  f.onResize(draw);
  update();
}

/** 칸 크기: 넓으면 [S | A | V·O] 한 줄, 좁으면 [S | A] / [V·O] 두 줄 */
function layout(W: number, H: number) {
  const stacked = W < 460;
  const c = stacked
    ? Math.floor(Math.min((W - 2 * pad - lab - gap) / (2 * NT), (H - 2 * head - 3 * pad) / (2 * NT + 1.5)))
    : Math.floor(Math.min((W - 2 * pad - lab - 2 * gap - wlab) / (2 * NT + DV), (H - head - 2 * pad - 20) / (NT + 1.6)));
  return { stacked, c };
}
