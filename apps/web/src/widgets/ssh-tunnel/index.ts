// SSH 포트 포워딩 방향 (data/widgets.toml: ssh-tunnel).
// 2.3장 "포트 포워딩"의 두 방향을 내 노트북 → bastion(ProxyJump) → gpu1 그림으로 보인다.
// -L 로컬포트:localhost:원격포트 — 내 노트북의 로컬포트로 들어온 연결을 gpu1이 본 localhost:원격포트(= gpu1 자신)로 넘긴다.
//    브라우저가 열 주소는 http://localhost:로컬포트다. 원격포트에 듣는 서비스가 없으면 연결이 거부된다.
// -R 원격포트:localhost:로컬포트 — gpu1의 원격포트로 들어온 연결을 내 쪽 localhost:로컬포트로 넘긴다.
// 서비스 포트는 본문 예(TensorBoard 6006, Jupyter 8888)와 본문 '함정'의 "다른 포트로 뜬" 경우(6007)를 쓴다.
// -R 쪽의 내 노트북 서비스(8080)는 설명을 위한 가상 예다.
import { frame, param, presets, readouts, el, C } from "../frame";

export const PORTS = [6006, 6007, 8080, 8888, 8890, 16006];
type Dir = "L" | "R";
type Svc = "tb" | "tbmoved" | "jupyter";
const SVC: Record<Svc, { label: string; name: string; port: number }> = {
  tb: { label: "TensorBoard", name: "TensorBoard", port: 6006 },
  tbmoved: { label: "6007로 뜸", name: "TensorBoard", port: 6007 },
  jupyter: { label: "Jupyter", name: "Jupyter", port: 8888 },
};
export const LOCAL_SVC = { name: "패키지 저장소 (가상)", port: 8080 };

/** 포워딩 명령, 접속할 주소, 성공 여부 */
export function forward(dir: Dir, lp: number, rp: number, svc: Svc) {
  if (dir === "L") {
    const ok = rp === SVC[svc].port;
    return { cmd: `ssh -N -L ${lp}:localhost:${rp} gpu1`, open: `http://localhost:${lp}`, where: "내 브라우저", ok, target: ok ? `${SVC[svc].name} 응답` : `연결 거부: gpu1의 ${rp}번에서 듣는 서비스 없음` };
  }
  const ok = lp === LOCAL_SVC.port;
  return { cmd: `ssh -N -R ${rp}:localhost:${lp} gpu1`, open: `curl localhost:${rp}`, where: "gpu1에서", ok, target: ok ? "내 쪽 패키지 저장소 응답" : `연결 거부: 내 노트북의 ${lp}번에서 듣는 서비스 없음` };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "SSH 포트 포워딩 방향", "manipulate", narrow ? 1.15 : 0.5, "내 노트북, bastion, gpu1 사이의 SSH 연결과 포워딩된 연결의 방향");
  let dir: Dir = "L", svc: Svc = "tb", lp = 16006, rp = 6006;

  type Box = { x: number; y: number; w: number; h: number };
  const rbox = (ctx: CanvasRenderingContext2D, b: Box, stroke: string, dash: number[] = [], lw = 1.2) => {
    ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.setLineDash(dash);
    ctx.beginPath(); ctx.roundRect(b.x, b.y, b.w, b.h, 8); ctx.stroke(); ctx.setLineDash([]);
  };
  const arrow = (ctx: CanvasRenderingContext2D, pts: [number, number][], col: string) => {
    ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.lineJoin = "round";
    ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    const [x1, y1] = pts[pts.length - 2], [x2, y2] = pts[pts.length - 1];
    const a = Math.atan2(y2 - y1, x2 - x1);
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - 10 * Math.cos(a - 0.4), y2 - 10 * Math.sin(a - 0.4)); ctx.lineTo(x2 - 10 * Math.cos(a + 0.4), y2 - 10 * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
  };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.textBaseline = "middle";
    const r = forward(dir, lp, rp, svc);
    ctx.fillStyle = C.mid; ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText(dir === "L" ? "-L: 내 쪽 포트 → gpu1이 본 목적지" : "-R: gpu1의 포트 → 내 쪽 목적지", 12, 14);
    // 상자 셋. 넓으면 가로(노트북 | bastion | gpu1), 좁으면 세로로 놓고, SSH 통로가 bastion을 지난다
    let A: Box, B: Box, G: Box, PA: [number, number], PG: [number, number];
    const ew = narrow ? Math.min(150, W * 0.44) : Math.min(170, W * 0.3 - 24), eh = 40;
    if (!narrow) {
      const bw = Math.min(210, W * 0.3), top = 30, h = H - top - 12;
      A = { x: 12, y: top, w: bw, h }; G = { x: W - 12 - bw, y: top, w: bw, h };
      const yb = top + h * 0.58;
      B = { x: W / 2 - 52, y: yb - 34, w: 104, h: 68 };
      PA = [A.x + A.w, yb]; PG = [G.x, yb];
    } else {
      const top = 28, bh = (H - top - 12) * 0.36;
      A = { x: 12, y: top, w: W - 24, h: bh }; G = { x: 12, y: H - 12 - bh, w: W - 24, h: bh };
      const xb = W - 12 - ew / 2 - 10;
      B = { x: xb - 100, y: (A.y + A.h + G.y) / 2 - 22, w: 116, h: 44 };
      PA = [xb, A.y + A.h]; PG = [xb, G.y];
    }
    const head = (b: Box, t: string, sub: string) => {
      rbox(ctx, b, C.faint);
      ctx.fillStyle = C.ink; ctx.font = "700 13px Pretendard Variable, sans-serif"; ctx.fillText(t, b.x + 10, b.y + 15);
      ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace"; ctx.fillText(sub, b.x + 10, b.y + 31, b.w - 20);
    };
    head(A, "내 노트북", "localhost = 내 노트북");
    head(G, "gpu1", "localhost = gpu1 자신");
    // SSH 연결(암호화된 통로)과 bastion
    rbox(ctx, B, C.faint, [4, 3]);
    ctx.fillStyle = C.mid; ctx.font = "700 12px Pretendard Variable, sans-serif";
    if (!narrow) { ctx.fillText("bastion", B.x + 10, B.y + 13); ctx.font = "11px Pretendard Variable, sans-serif"; ctx.fillText("통로만 (ProxyJump)", B.x + 10, B.y + B.h - 11, B.w - 16); }
    else { ctx.fillText("bastion", B.x + 8, B.y + 13); ctx.font = "11px Pretendard Variable, sans-serif"; ctx.fillText("ProxyJump", B.x + 8, B.y + B.h - 11); }
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 10; ctx.globalAlpha = 0.3;
    ctx.beginPath(); ctx.moveTo(...PA); ctx.lineTo(...PG); ctx.stroke(); ctx.globalAlpha = 1;
    if (!narrow && B.x - PA[0] > 54) { ctx.fillStyle = C.ink2; ctx.font = "700 11px JetBrains Mono, monospace"; ctx.fillText("SSH", (PA[0] + B.x) / 2 - 10, PA[1] - 14); }

    const tag = (b: Box, t: string, sub: string, col: string, dash: number[] = []) => {
      ctx.fillStyle = C.bg; ctx.fillRect(b.x, b.y, b.w, b.h);
      rbox(ctx, b, col, dash, 1.6);
      ctx.fillStyle = col; ctx.font = "700 12px Pretendard Variable, sans-serif"; ctx.fillText(t, b.x + 8, b.y + 13, b.w - 30);
      ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace"; ctx.fillText(sub, b.x + 8, b.y + 29, b.w - 14);
    };
    // 연결을 시작하는 쪽(src 상자)은 위쪽 모서리, 받는 쪽(dst 상자)은 통로 끝 맞은편
    const [S, Dbox] = dir === "L" ? [A, G] : [G, A];
    const [Pin, Pout] = dir === "L" ? [PA, PG] : [PG, PA];
    const src: Box = narrow ? { x: S.x + 10, y: S.y + 42, w: ew, h: eh } : { x: S.x + 12, y: S.y + 44, w: S.w - 24, h: eh };
    const dst: Box = narrow
      ? { x: Pout[0] - ew / 2, y: Dbox.y + 42, w: ew, h: eh }
      : { x: Dbox.x + 12, y: Pout[1] - eh / 2, w: Dbox.w - 24, h: eh };
    const alt: Box = narrow ? { x: Dbox.x + 10, y: Dbox.y + 42, w: Math.min(ew, dst.x - Dbox.x - 20), h: eh } : { ...dst, y: dst.y + eh + 12 };
    const service = dir === "L" ? { name: SVC[svc].name, port: SVC[svc].port } : LOCAL_SVC;
    const want = dir === "L" ? rp : lp;
    tag(src, dir === "L" ? "브라우저" : "pip·curl", dir === "L" ? `localhost:${lp}` : `localhost:${rp}`, C.ink);
    if (r.ok) tag(dst, service.name, `localhost:${service.port}`, C.good);
    else { tag(dst, `빈 포트 ${want}`, "듣는 프로세스 없음", C.mid, [4, 3]); tag(alt, service.name, `localhost:${service.port}`, C.good); }
    // 포워딩된 연결: 시작 → 통로 입구(ssh가 듣는 포트) → 통로 → 목적지
    const entry: [number, number] = narrow ? [src.x + src.w, src.y + src.h / 2] : [src.x + src.w / 2, src.y + src.h];
    const into: [number, number] = narrow ? [Pout[0], dir === "L" ? dst.y : dst.y + dst.h] : dir === "L" ? [dst.x, Pout[1]] : [dst.x + dst.w, Pout[1]];
    arrow(ctx, [entry, narrow ? [Pin[0], entry[1]] : [entry[0], Pin[1]], Pin], C.edge);
    arrow(ctx, [Pin, Pout], C.edge);
    arrow(ctx, [Pout, into], r.ok ? C.edge : C.mid);
    ctx.fillStyle = C.bg; ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(Pin[0], Pin[1], 6, 0, 7); ctx.fill(); ctx.stroke();
    ctx.fillStyle = C.edge; ctx.font = "700 12px JetBrains Mono, monospace";
    const pl = `:${dir === "L" ? lp : rp}`, pw = ctx.measureText(pl).width;
    if (narrow) ctx.fillText(pl, Pin[0] - pw - 10, Pin[1] + (dir === "L" ? 12 : -12));
    else ctx.fillText(pl, dir === "L" ? Pin[0] - pw - 8 : Pin[0] + 8, Pin[1] + 16);
    // 결과 표시: 목적지 상자 오른쪽 위에 ✓ 또는 ✕
    const ex = dst.x + dst.w - 14, ey = dst.y + 13;
    ctx.lineWidth = 2.6;
    if (r.ok) { ctx.strokeStyle = C.good; ctx.beginPath(); ctx.moveTo(ex - 6, ey); ctx.lineTo(ex - 1, ey + 5); ctx.lineTo(ex + 7, ey - 5); ctx.stroke(); }
    else { ctx.strokeStyle = C.edge; ctx.beginPath(); ctx.moveTo(ex - 5, ey - 5); ctx.lineTo(ex + 5, ey + 5); ctx.moveTo(ex + 5, ey - 5); ctx.lineTo(ex - 5, ey + 5); ctx.stroke(); }
  }

  function update() {
    const r = forward(dir, lp, rp, svc);
    ro.set("cmd", r.cmd, "hot");
    ro.set("open", `${r.where}: ${r.open}`);
    ro.set("res", r.target, r.ok ? "ok" : "");
    svcRow.hidden = dir !== "L";
    draw();
  }

  const dirBtns = presets<Dir>([{ key: "L", label: "-L 로컬 포워딩" }, { key: "R", label: "-R 원격 포워딩" }], (k) => {
    dir = k;
    if (k === "L") { lp = 16006; rp = 6006; } else { rp = 8080; lp = LOCAL_SVC.port; }
    pl.set(PORTS.indexOf(lp)); pr.set(PORTS.indexOf(rp)); update();
  }, dir);
  dirBtns.row.classList.add("segmented");
  const short: Record<Svc, string> = { tb: "TB 6006", tbmoved: "TB 6007", jupyter: "Jupyter" };
  const svcBtns = presets<Svc>((Object.keys(SVC) as Svc[]).map((k) => ({ key: k, label: narrow ? short[k] : SVC[k].label })), (k) => { svc = k; update(); }, svc);
  const svcRow = svcBtns.row;
  const pl = param({ label: "로컬", min: 0, max: PORTS.length - 1, step: 1, value: PORTS.indexOf(lp), fmt: (i) => String(PORTS[i]), aria: "내 노트북 쪽 포트", onInput: (i) => { lp = PORTS[i]; update(); } });
  const pr = param({ label: "원격", min: 0, max: PORTS.length - 1, step: 1, value: PORTS.indexOf(rp), fmt: (i) => String(PORTS[i]), aria: "gpu1 쪽 포트", onInput: (i) => { rp = PORTS[i]; update(); } });
  for (const p of [pl, pr]) p.row.style.gridTemplateColumns = "2.6em minmax(0, 1fr) 3.6em";
  const grid = el("div", "param-grid");
  grid.append(pl.row, pr.row);
  const ro = readouts();
  f.panel.append(dirBtns.row, svcRow, grid, ro.row);
  f.onResize(draw);
  update();
}
