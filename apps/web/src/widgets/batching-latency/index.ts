// 배치 크기와 지연·처리량 (data/widgets.toml: batching-latency).
// 22.7장 "batch 크기의 거래"의 비용 모형: batch 하나의 계산 시간 t(B) = t0 + B·c, t0 = 4 ms, c = 0.05 ms(가정).
// 동적 batching 서버를 요청 하나하나로 시뮬레이션한다: 서버가 비면 대기열의 첫 요청부터 최대 B개를 모으되,
// B개가 차거나 첫 요청이 최대 대기 시간 T만큼 기다리면 그때 있는 요청을 한 batch로 처리한다.
// 도착이 고르면(본문의 가정) 평균 지연이 본문 표의 t(B) + (B−1)/(2λ)와 같고, 무작위(Poisson) 도착이면 꼬리(p99)가 길어진다.
import { frame, param, readouts, el, C, alpha, rng } from "../frame";

export const T0 = 4.0, CPS = 0.05;                         // ms
const NREQ = 24000;
const BS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
const INF_WAIT = 250;                                      // 슬라이더 끝 = 가득 찰 때까지 기다림

export function arrivals(lambda: number, poisson: boolean, n = NREQ) {
  // B = 512에서도 정상 상태의 batch가 충분히 많도록 요청 수를 넉넉히 둔다
  const a = new Float64Array(n), r = rng(12);
  let t = 0;
  for (let i = 0; i < n; i++) { t += poisson ? -Math.log(1 - r.u()) * 1000 / lambda : 1000 / lambda; a[i] = t; }
  return a;
}

/** 요청별 지연(ms)과 batch 기록. wait = 첫 요청이 기다릴 수 있는 최대 시간(ms, Infinity면 B개가 찰 때까지). */
export function simulate(a: Float64Array, B: number, wait: number) {
  const n = a.length, lat = new Float64Array(n);
  const batches: { start: number; done: number; from: number; count: number }[] = [];
  let free = 0, q = 0;
  while (q < n) {
    const tFull = q + B - 1 < n ? a[q + B - 1] : a[n - 1];
    const start = Math.max(free, Math.min(tFull, a[q] + wait));
    let count = 0;
    while (count < B && q + count < n && a[q + count] <= start + 1e-9) count++;
    const done = start + T0 + count * CPS;
    for (let i = q; i < q + count; i++) lat[i] = done - a[i];
    batches.push({ start, done, from: q, count });
    free = done; q += count;
  }
  // 시작·끝 효과를 빼려고 앞뒤 10%에 걸친 batch는 빼고 잰다
  const s: number[] = [];
  for (const b of batches) if (b.from >= n * 0.1 && b.from + b.count <= n * 0.9) for (let i = b.from; i < b.from + b.count; i++) s.push(lat[i]);
  s.sort((x, y) => x - y);
  const pct = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  const mean = s.reduce((x, y) => x + y, 0) / s.length;
  const thr = (n * 1000) / (batches[batches.length - 1].done - a[0]);
  const capacity = (B * 1000) / (T0 + B * CPS);
  return { lat, batches, p50: pct(0.5), p99: pct(0.99), mean, throughput: thr, capacity, meanBatch: n / batches.length };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "배치 크기와 지연·처리량", "manipulate", narrow ? 1.0 : 0.58, "요청 도착과 batch 처리의 시간선, 그리고 최대 batch 크기마다의 p50·p99 지연");
  let lambda = 2000, bi = 7, waitMs = INF_WAIT, poisson = false;
  let arr = arrivals(lambda, poisson);
  const waitOf = () => (waitMs >= INF_WAIT ? Infinity : waitMs);
  const overloaded = (B: number) => lambda >= 0.98 * (B * 1000) / (T0 + B * CPS);
  type Sim = ReturnType<typeof simulate> & { b: number; over: boolean };
  let sims: Sim[] = [];

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const L = 44, R = 14, top = 22, split = H * 0.4, bottom = H - 28;
    const sim = sims[bi];
    ctx.textBaseline = "alphabetic";
    // 위: 시간선. 정상 상태의 batch 3개를 보인다(앞 절반 지점에서 시작)
    const k0 = sim.batches.findIndex((b) => b.from >= NREQ / 2);
    const shown = sim.batches.slice(k0, k0 + 3);
    const t0 = arr[shown[0].from] - 1, t1 = Math.max(shown[shown.length - 1].done, t0 + 4) + 1;
    const X = (t: number) => L + ((t - t0) / (t1 - t0)) * (W - L - R);
    const yA = top + 18, yS = split - 30;
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
    ctx.fillText("요청", 4, yA + 4); ctx.fillText("서버", 4, yS + 8);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, yA); ctx.lineTo(W - R, yA); ctx.moveTo(L, yS + 16); ctx.lineTo(W - R, yS + 16); ctx.stroke();
    const cols = [C.ink2, C.mid, C.ink2];
    shown.forEach((b, j) => {
      const col = j === 1 ? C.edge : cols[j];
      ctx.strokeStyle = alpha(col, 0.35); ctx.lineWidth = 1;
      for (let i = b.from; i < b.from + b.count; i++) {
        if (arr[i] < t0) continue;
        const x = X(arr[i]);
        ctx.beginPath(); ctx.moveTo(x, yA); ctx.lineTo(X(b.start), yS); ctx.stroke();
      }
      ctx.fillStyle = col;
      for (let i = b.from; i < b.from + b.count; i++) { if (arr[i] < t0) continue; ctx.fillRect(X(arr[i]) - 1, yA - 7, 2, 14); }
      ctx.fillRect(X(b.start), yS, Math.max(2, X(b.done) - X(b.start)), 16);
    });
    const mid = shown[1] ?? shown[0];
    ctx.fillStyle = C.edge; ctx.font = "600 11px Pretendard Variable, sans-serif";
    const bl = `batch ${mid.count}개 · ${(T0 + mid.count * CPS).toFixed(1)} ms`;
    ctx.fillText(bl, Math.min(Math.max(L, X(mid.start)), W - R - ctx.measureText(bl).width), yS + 30);
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    const span = `${(t1 - t0).toFixed(t1 - t0 < 20 ? 1 : 0)} ms`;
    ctx.fillText(span, W - R - ctx.measureText(span).width, top + 2);

    // 아래: 최대 batch마다 p50·p99(로그 가로축), 과부하인 B는 칠한다
    const stable = sims.filter((s) => !s.over);
    const yMax = Math.max(20, ...stable.map((s) => s.p99)) * 1.12;
    const top2 = split + 18;
    const X2 = (i: number) => L + (i / (BS.length - 1)) * (W - L - R);
    const Y2 = (v: number) => bottom - (Math.min(v, yMax) / yMax) * (bottom - top2);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, bottom); ctx.lineTo(W - R, bottom); ctx.moveTo(L, top2); ctx.lineTo(L, bottom); ctx.stroke();
    const step = [5, 10, 20, 50, 100, 200, 500, 1000].find((s) => yMax / s <= 4.5) ?? 2000;
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (let v = 0; v <= yMax; v += step) {
      ctx.fillText(String(v), L - 6 - ctx.measureText(String(v)).width, Y2(v) + 4);
      if (v) { ctx.strokeStyle = C.faint; ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(L, Y2(v)); ctx.lineTo(W - R, Y2(v)); ctx.stroke(); ctx.setLineDash([]); }
    }
    BS.forEach((b, i) => { if (narrow && i % 2 && i !== BS.length - 1) return; const s = String(b); ctx.fillText(s, X2(i) - ctx.measureText(s).width / 2, bottom + 15); });
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText("지연(ms) — 최대 batch B →", L + 4, top2 - 6);
    const dx = (W - L - R) / (BS.length - 1);
    sims.forEach((s, i) => { if (s.over) { ctx.fillStyle = alpha(C.mid, 0.16); ctx.fillRect(X2(i) - dx / 2, top2, dx, bottom - top2); } });
    if (sims.some((s) => s.over)) {
      const i0 = sims.findIndex((s) => s.over);
      ctx.fillStyle = C.mid; ctx.font = "600 11px Pretendard Variable, sans-serif";
      ctx.fillText("과부하", Math.max(L + 2, X2(i0) - dx / 2 + 3), bottom - 6);
    }
    const series = (key: "p50" | "p99", col: string, w: number, dash: number[]) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash);
      ctx.beginPath(); let on = false;
      sims.forEach((s, i) => { if (s.over) { on = false; return; } const x = X2(i), y = Y2(s[key]); if (on) ctx.lineTo(x, y); else ctx.moveTo(x, y); on = true; });
      ctx.stroke(); ctx.setLineDash([]);
      sims.forEach((s, i) => { if (s.over) return; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X2(i), Y2(s[key]), 3, 0, 7); ctx.fill(); });
    };
    series("p50", C.ink2, 2, []);
    series("p99", C.ink, 1.6, [5, 4]);
    const cur = sims[bi];
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X2(bi), top2); ctx.lineTo(X2(bi), bottom); ctx.stroke();
    if (!cur.over) { ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(X2(bi), Y2(cur.p99), 5, 0, 7); ctx.arc(X2(bi), Y2(cur.p50), 5, 0, 7); ctx.fill(); }
    // 범례
    ctx.font = "11px JetBrains Mono, monospace";
    const lx = W - R - 96;
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(lx, top2 + 8); ctx.lineTo(lx + 16, top2 + 8); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.fillText("p50", lx + 20, top2 + 12);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(lx + 50, top2 + 8); ctx.lineTo(lx + 66, top2 + 8); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText("p99", lx + 70, top2 + 12);
  }

  // B만 바꾸면 다시 시뮬레이션하지 않는다
  function ui() { update(false, false); }
  function update(regen = false, resim = true) {
    if (regen) arr = arrivals(lambda, poisson);
    if (resim || !sims.length) sims = BS.map((b) => ({ b, over: overloaded(b), ...simulate(arr, b, waitOf()) }));
    const B = BS[bi], s = sims[bi];
    if (overloaded(B)) {
      ro.set("lat", `과부하: 용량 ${s.capacity.toFixed(0)}/s < 도착 ${lambda}/s`, "hot");
      ro.set("thr", "대기열이 끝없이 길어진다");
      ro.set("batch", `이용률 ${(lambda / s.capacity * 100).toFixed(0)}%`);
    } else {
      ro.set("lat", `p50 ${s.p50.toFixed(1)} ms · p99 ${s.p99.toFixed(1)} ms`, "hot");
      ro.set("thr", `처리량 ${s.throughput.toFixed(0)}/s · 용량 ${s.capacity.toFixed(0)}/s`);
      ro.set("batch", `평균 batch ${s.meanBatch.toFixed(1)} · 평균 지연 ${s.mean.toFixed(2)} ms`);
    }
    draw();
  }

  const pl = param({ label: "λ", min: 100, max: 10000, step: 100, value: lambda, fmt: (v) => `${v}/s`, aria: "요청 도착률 λ(초당 요청 수)", onInput: (v) => { lambda = v; update(true); } });
  const pb = param({ label: "B", min: 0, max: BS.length - 1, step: 1, value: bi, fmt: (v) => String(BS[v]), aria: "최대 batch 크기 B (2의 거듭제곱)", onInput: (v) => { bi = v; ui(); } });
  const pw = param({ label: "T", min: 0, max: INF_WAIT, step: 5, value: waitMs, fmt: (v) => (v >= INF_WAIT ? "∞" : `${v} ms`), aria: "첫 요청의 최대 대기 시간 T(끝 = 가득 찰 때까지)", onInput: (v) => { waitMs = v; update(); } });
  for (const p of [pl, pb, pw]) p.row.style.gridTemplateColumns = "1.6em minmax(0, 1fr) 4.6em";
  const pb2 = pb.input; pb2.setAttribute("aria-valuetext", `B = ${BS[bi]}`);
  pb2.addEventListener("input", () => pb2.setAttribute("aria-valuetext", `B = ${BS[bi]}`));
  const grid = el("div", "param-grid");
  grid.append(pl.row, pb.row, pw.row);
  const tools = el("div", "wframe-row");
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" /> 무작위 도착(Poisson)`;
  sw.querySelector("input")!.addEventListener("change", (e) => { poisson = (e.target as HTMLInputElement).checked; update(true); });
  tools.append(sw);
  const ro = readouts();
  f.panel.append(grid, tools, ro.row);
  f.onResize(draw);
  update();
}
