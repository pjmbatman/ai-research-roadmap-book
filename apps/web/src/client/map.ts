// 로드맵 지도: 트랙을 레이어 순서로 열에 놓고, 장을 점으로, 선수 지식을 곡선으로 그린다(canvas).
// 가리킨 장의 선수 지식 전체(조상)를 코랄로, 그 장에 의존하는 장(자손)을 밝게 강조한다.
import { progress, chapterScore } from "./store";

interface N { id: string; label: string; title: string; url: string; tier: string; status: string; track: string; pre: string[]; goals: string[]; quiz: number; ex: number; checks: number; x: number; y: number }
const READY = new Set(["draft", "review", "done"]);

export function mountMap() {
  const data = JSON.parse(document.getElementById("map-data")!.textContent!);
  const root = document.querySelector<HTMLElement>("[data-map]")!;
  const canvas = root.querySelector("canvas")!;
  const info = root.querySelector<HTMLElement>("[data-map-info]")!;
  const ctx = canvas.getContext("2d")!;
  const base = document.body.dataset.base || "/";
  const tracks: { key: string; code: string; name: string; layer: number }[] = data.tracks;
  const nodes: N[] = data.nodes;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const kids = new Map<string, string[]>();
  nodes.forEach((n) => n.pre.forEach((p) => kids.set(p, [...(kids.get(p) ?? []), n.id])));
  const closure = (id: string, next: (id: string) => string[]) => {
    const seen = new Set<string>(); const st = [...next(id)];
    while (st.length) { const k = st.pop()!; if (seen.has(k)) continue; seen.add(k); st.push(...next(k)); }
    return seen;
  };
  let filter = "all";
  let hover: N | null = null, pinned: N | null = null;
  let W = 0, H = 0;
  const visited = () => progress.all();
  // 진도 상태(benchmarks.md §3.5): 시작 전 · 학습 중(방문) · 완료(퀴즈·연습문제·통과 기준을 모두 마침)
  const stateOf = (n: N): "none" | "learning" | "done" => {
    if (!visited()[n.id]?.visited) return "none";
    const sc = chapterScore(n.id, { quiz: n.quiz, exercises: n.ex, checks: n.checks });
    return sc.total && sc.ratio >= 1 ? "done" : "learning";
  };
  const STATE_LABEL = { none: "시작 전", learning: "학습 중", done: "완료" };

  function layout() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const colW = (W - 40) / tracks.length;
    const top = 92, bottom = H - 24;
    tracks.forEach((t, ci) => {
      const col = nodes.filter((n) => n.track === t.key);
      const gap = Math.min(22, (bottom - top) / Math.max(col.length, 1));
      col.forEach((n, k) => { n.x = 20 + colW * (ci + 0.5); n.y = top + gap * (k + 0.5); });
    });
    draw();
  }

  const shown = (n: N) => filter === "all" || (filter === "core" && n.tier === "core") || (filter === "ready" && READY.has(n.status)) || (filter === "visited" && !!visited()[n.id]?.visited);

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const colW = (W - 40) / tracks.length;
    // 레이어 띠와 열 머리
    let prevLayer = -1;
    tracks.forEach((t, ci) => {
      const x = 20 + colW * ci;
      if (t.layer !== prevLayer) {
        ctx.fillStyle = "rgba(220,230,242,0.04)";
        if (t.layer % 2 === 0) { const span = tracks.filter((u) => u.layer === t.layer).length; ctx.fillRect(x, 54, colW * span, H - 64); }
        ctx.fillStyle = "rgba(220,230,242,0.45)"; ctx.font = "10.5px JetBrains Mono, monospace";
        ctx.fillText(`L${t.layer}`, x + 4, 66);
        prevLayer = t.layer;
      }
      ctx.fillStyle = "#dce6f2"; ctx.font = "600 12px JetBrains Mono, monospace";
      const w = ctx.measureText(t.code).width;
      ctx.fillText(t.code, x + colW / 2 - w / 2, 82);
    });
    const focus = pinned ?? hover;
    const anc = focus ? closure(focus.id, (id) => byId.get(id)?.pre ?? []) : new Set<string>();
    const desc = focus ? closure(focus.id, (id) => kids.get(id) ?? []) : new Set<string>();
    // 간선
    for (const n of nodes) for (const p of n.pre) {
      const a = byId.get(p); if (!a) continue;
      const hot = focus && (n.id === focus.id || anc.has(n.id)) && (anc.has(p));
      const cool = focus && (p === focus.id || desc.has(p)) && (desc.has(n.id));
      if (focus && !hot && !cool) { ctx.strokeStyle = "rgba(220,230,242,0.035)"; }
      else if (hot) { ctx.strokeStyle = "rgba(255,107,90,0.85)"; }
      else if (cool) { ctx.strokeStyle = "rgba(220,230,242,0.55)"; }
      else ctx.strokeStyle = shown(n) && shown(a) ? "rgba(220,230,242,0.09)" : "rgba(220,230,242,0.03)";
      ctx.lineWidth = hot || cool ? 1.4 : 1;
      ctx.beginPath(); ctx.moveTo(a.x, a.y);
      const mx = (a.x + n.x) / 2;
      ctx.bezierCurveTo(mx, a.y, mx, n.y, n.x, n.y); ctx.stroke();
    }
    // 노드: 채움 = 진도 상태, 코랄 = 지금 보는 장과 그 선수 지식
    for (const n of nodes) {
      const on = shown(n);
      const isF = focus?.id === n.id, isA = anc.has(n.id), isD = desc.has(n.id);
      const st = stateOf(n);
      const r = isF ? 6.5 : n.tier === "core" ? 4 : 3;
      const dim = (focus && !isF && !isA && !isD) || !on;
      ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 7);
      ctx.fillStyle = isF || isA ? "#ff6b5a" : dim ? "rgba(220,230,242,0.14)" : st === "done" ? "#3fc79a" : st === "learning" ? "#dce6f2" : READY.has(n.status) ? "rgba(220,230,242,0.7)" : "rgba(220,230,242,0.38)";
      ctx.fill();
      if (st === "learning" && !dim && !isF && !isA) { ctx.strokeStyle = "#3fc79a"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(n.x, n.y, r + 2.5, 0, 7); ctx.stroke(); }
      if (isF) { ctx.fillStyle = "#dce6f2"; ctx.font = "600 12px Pretendard Variable, sans-serif"; const s = `${n.label} ${n.title}`; const w = ctx.measureText(s).width; ctx.fillText(s, Math.min(n.x + 10, W - w - 8), n.y - 9); }
    }
  }

  function showInfo(n: N | null) {
    info.hidden = !n;
    if (!n) return;
    const t = tracks.find((x) => x.key === n.track);
    const anc = closure(n.id, (id) => byId.get(id)?.pre ?? []);
    const desc = closure(n.id, (id) => kids.get(id) ?? []);
    const st = stateOf(n);
    info.innerHTML = `<div class="mi-top"><span class="badge"></span><span class="mi-state"></span></div><b></b><ul class="mi-goals"></ul><div class="nums"><span><i class="hot"></i></span><span><i></i></span></div><a class="btn btn-primary"></a>`;
    info.querySelector(".badge")!.textContent = t?.code ?? "";
    const stEl = info.querySelector<HTMLElement>(".mi-state")!;
    stEl.textContent = STATE_LABEL[st];
    stEl.dataset.state = st;
    info.querySelector("b")!.textContent = `${n.label} ${n.title}`;
    const ul = info.querySelector(".mi-goals")!;
    n.goals.forEach((g) => { const li = document.createElement("li"); li.textContent = g; ul.append(li); });
    const nums = info.querySelectorAll(".nums > span");
    nums[0].append(`선수 ${anc.size}`);
    nums[1].append(`다음 ${desc.size}`);
    const a = info.querySelector<HTMLAnchorElement>("a.btn")!;
    a.href = base + n.url;
    a.textContent = st === "none" ? "읽기 시작 →" : "이어서 읽기 →";
  }

  const at = (e: MouseEvent) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    let best: N | null = null, d = 12;
    for (const n of nodes) { const dd = Math.hypot(n.x - x, n.y - y); if (dd < d && shown(n)) { d = dd; best = n; } }
    return best;
  };
  canvas.addEventListener("pointermove", (e) => { const n = at(e); canvas.style.cursor = n ? "pointer" : "default"; if (n !== hover) { hover = n; if (!pinned) showInfo(n); draw(); } });
  canvas.addEventListener("pointerleave", () => { hover = null; if (!pinned) showInfo(null); draw(); });
  canvas.addEventListener("click", (e) => { const n = at(e as PointerEvent); pinned = n && pinned?.id !== n.id ? n : null; showInfo(pinned ?? n); draw(); });
  root.querySelectorAll<HTMLButtonElement>("[data-filter]").forEach((b) => b.addEventListener("click", () => {
    filter = b.dataset.filter!;
    root.querySelectorAll("[data-filter]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    draw();
  }));
  new ResizeObserver(layout).observe(canvas);
  showInfo(null);
  layout();
}
