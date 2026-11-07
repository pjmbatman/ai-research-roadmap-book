// z와 robust z (data/widgets.toml: robust-outlier).
// 15.3장의 예: 정상값 90개 ~ N(10, 1)에 이상치 k개 ~ U(m − 10, m + 10)를 섞는다(본문은 k = 10, m = 50).
//   z = (x − x̄)/s (s는 np.std처럼 n으로 나눈다), |z| > 3을 이상치로 본다.
//   robust z = 0.6745 (x − median)/MAD, MAD = median |x − median|, |robust z| > 3.5를 이상치 후보로 본다.
// 이상치가 많거나 클수록 s가 부풀어 z의 문턱 구간(x̄ ± 3s)이 이상치를 삼킨다(masking). 중앙값과 MAD는 거의 흔들리지 않는다.
import { frame, param, readouts, el, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

export const N0 = 90, KMAX = 40;
const BASE = (() => {
  const r = rng(315);
  const normal = Array.from({ length: N0 }, () => 10 + r.normal());
  const u = Array.from({ length: KMAX }, () => r.u());
  const jit = Array.from({ length: N0 + KMAX }, () => r.u());
  return { normal, u, jit };
})();

const median = (a: number[]) => { const s = [...a].sort((p, q) => p - q), n = s.length; return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]); };

/** 정상값 90개 + 이상치 k개(중심 m)의 z·robust z 검출 결과 */
export function analyze(k: number, m: number) {
  const out = BASE.u.slice(0, k).map((u) => m - 10 + 20 * u);
  const x = [...BASE.normal, ...out];
  const n = x.length;
  const mean = x.reduce((s, v) => s + v, 0) / n;
  const sd = Math.sqrt(x.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  const med = median(x);
  const mad = median(x.map((v) => Math.abs(v - med)));
  const z = x.map((v) => (v - mean) / sd);
  const rz = x.map((v) => (0.6745 * (v - med)) / mad);
  const isOut = x.map((_, i) => i >= N0);
  const count = (flag: boolean[]) => ({ hit: flag.filter((f, i) => f && isOut[i]).length, false: flag.filter((f, i) => f && !isOut[i]).length });
  const zFlag = z.map((v) => Math.abs(v) > 3), rFlag = rz.map((v) => Math.abs(v) > 3.5);
  return { x, mean, sd, med, mad, zFlag, rFlag, z: count(zFlag), r: count(rFlag), k };
}

export function mount(root: HTMLElement) {
  const f = frame(root, "z와 robust z", "manipulate", 0.46, "정상값과 이상치를 수직선에 찍고, 평균 ± 3 표준편차 구간과 중앙값 ± 3.5 robust 척도 구간 밖에서 검출된 점을 표시한 그림");
  let k = 10, m = 50;
  let res = analyze(k, m);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const PL = 14, PR = 14, top = 10, axisY = H - 22;
    const lo = 0, hi = Math.max(70, m + 14);
    const X = (v: number) => PL + ((v - lo) / (hi - lo)) * (W - PL - PR);
    const rowH = (axisY - top - 8) / 2;
    const rows = [
      { name: "z = (x − 평균)/s,  |z| > 3", c: res.mean, half: 3 * res.sd, flag: res.zFlag, mark: "평균" },
      { name: "robust z,  |z| > 3.5", c: res.med, half: (3.5 * res.mad) / 0.6745, flag: res.rFlag, mark: "중앙값" },
    ];
    ctx.font = "11px JetBrains Mono, monospace";
    rows.forEach((row, ri) => {
      const y0 = top + ri * (rowH + 8), y1 = y0 + rowH, mid = (y0 + y1) / 2 + 6;
      // 문턱 구간(이 안은 정상으로 본다)
      const a = Math.max(X(row.c - row.half), PL), b = Math.min(X(row.c + row.half), W - PR);
      ctx.fillStyle = alpha(C.ink2, 0.16);
      ctx.fillRect(a, y0 + 16, b - a, rowH - 18);
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
      for (const xb of [row.c - row.half, row.c + row.half]) { const px = X(xb); if (px > PL && px < W - PR) { ctx.beginPath(); ctx.moveTo(px, y0 + 16); ctx.lineTo(px, y1); ctx.stroke(); } }
      ctx.setLineDash([]);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(X(row.c), y0 + 16); ctx.lineTo(X(row.c), y1); ctx.stroke();
      // 점: 검출된 점은 채운 코랄, 놓친 이상치는 빈 원, 정상값은 작은 회색 점
      res.x.forEach((v, i) => {
        const py = mid + (BASE.jit[i] - 0.5) * (rowH - 30);
        const out = i >= N0, hit = row.flag[i];
        ctx.beginPath(); ctx.arc(X(v), py, hit ? 3.6 : out ? 3.4 : 2.2, 0, 7);
        if (hit) { ctx.fillStyle = C.edge; ctx.fill(); }
        else if (out) { ctx.strokeStyle = C.edge; ctx.lineWidth = 1.4; ctx.stroke(); }
        else { ctx.fillStyle = alpha(C.mid, 0.7); ctx.fill(); }
      });
      ctx.fillStyle = C.ink; ctx.font = "600 12px Pretendard Variable, sans-serif";
      ctx.fillText(row.name, PL, y0 + 11);
      ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
      const lab = row.mark;
      const lx = Math.min(X(row.c) + 4, W - PR - ctx.measureText(lab).width);
      ctx.fillText(lab, lx, y1 - 3);
    });
    // 축
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PL, axisY); ctx.lineTo(W - PR, axisY); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    const step = hi > 80 ? 20 : 10;
    for (let v = 0; v <= hi; v += step) { const s = String(v); ctx.fillText(s, X(v) - ctx.measureText(s).width / 2, axisY + 14); }
  }

  let raf = 0;
  function update() {
    res = analyze(k, m);
    ro.set("data", `정상 90개 + 이상치 ${k}개 ~ U(${m - 10}, ${m + 10})`);
    ro.set("mean", `평균 ${res.mean.toFixed(1)} · 표준편차 ${res.sd.toFixed(1)}`);
    ro.set("med", `중앙값 ${res.med.toFixed(1)} · MAD ${res.mad.toFixed(2)}`);
    const zt = k ? `${res.z.hit}/${k}` : "0/0";
    ro.set("z", `|z| > 3: 이상치 ${zt} 검출${res.z.false ? ` · 정상값 ${res.z.false}개 오검출` : ""}`, k && res.z.hit < k ? "hot" : "");
    ro.set("r", `|robust z| > 3.5: 이상치 ${k ? `${res.r.hit}/${k}` : "0/0"} 검출${res.r.false ? ` · 정상값 ${res.r.false}개 오검출` : ""}`, k && res.r.hit === k ? "ok" : "");
    pageState.set("robust-outlier", { k, m, zHit: res.z.hit, rHit: res.r.hit }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pk = param({ label: "k", min: 0, max: KMAX, step: 1, value: k, fmt: (v) => `${v}개`, aria: "이상치 개수 (정상값 90개에 섞는다)", onInput: (v) => { k = v; update(); } });
  const pm = param({ label: "m", min: 20, max: 90, step: 1, value: m, fmt: (v) => `${v}`, aria: "이상치 중심값 m (이상치는 m − 10에서 m + 10 사이에 고르게 퍼진다)", onInput: (v) => { m = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pk.row, pm.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update();
}
