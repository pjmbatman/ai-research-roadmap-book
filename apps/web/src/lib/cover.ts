// 커버 이미지 (ADR 0011): 사진 대신 트랙 색 그라데이션 위에 장 id로 정해지는 추상 구도를 그린 SVG.
// 서버(빌드)와 브라우저 양쪽에서 쓰는 순수 함수다. 같은 입력은 언제나 같은 커버를 만든다.

/** 트랙별 두 색 (위 왼쪽 → 아래 오른쪽 그라데이션) */
export const TRACK_COLORS: Record<string, [string, string]> = {
  orientation: ["#6E56CF", "#3E63DD"],
  "research-infrastructure": ["#0D74CE", "#05A2C2"],
  "research-software-engineering": ["#12A594", "#2F7C57"],
  "mathematical-foundations": ["#3E63DD", "#8E4EC6"],
  "probabilistic-inference": ["#D6409F", "#8E4EC6"],
  "ml-core": ["#E54D2E", "#D6409F"],
  "ml-advanced": ["#F76B15", "#E54D2E"],
  "ml-systems": ["#5B5BD6", "#0D74CE"],
  "research-methodology": ["#AB6400", "#E54D2E"],
  "scientific-ml": ["#0090FF", "#12A594"],
  "domain-applications": ["#2B9A66", "#0090FF"],
  capstone: ["#1C2024", "#6E56CF"],
  reference: ["#60646C", "#1C2024"],
};

function hash(s: string) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function rand(seed: number) {
  let a = seed;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

export interface CoverInput { id: string; track: string; code: string; label?: string; ratio?: number }

/** 3:4(기본) 또는 지정한 비율의 커버 SVG 문자열 */
export function coverSVG({ id, track, code, label, ratio = 4 / 3 }: CoverInput): string {
  const W = 300, H = Math.round(W * ratio);
  const [c1, c2] = TRACK_COLORS[track] ?? TRACK_COLORS.reference;
  const h = hash(id);
  const r = rand(h);
  const gid = `g${h.toString(36)}`;
  const kind = h % 4;
  let art = "";
  if (kind === 0) {
    // 동심원
    const cx = 60 + r() * 180, cy = 70 + r() * (H - 180);
    for (let i = 6; i >= 1; i--) art += `<circle cx="${cx.toFixed(0)}" cy="${cy.toFixed(0)}" r="${i * 26}" fill="none" stroke="#fff" stroke-opacity="${(0.07 + i * 0.025).toFixed(3)}" stroke-width="${i % 2 ? 10 : 2}"/>`;
    art += `<circle cx="${cx.toFixed(0)}" cy="${cy.toFixed(0)}" r="16" fill="#fff" fill-opacity=".9"/>`;
  } else if (kind === 1) {
    // 점 격자 + 강조 점
    for (let y = 40; y < H - 90; y += 22) for (let x = 30; x < W - 10; x += 22) {
      const big = r() < 0.06;
      art += `<circle cx="${x}" cy="${y}" r="${big ? 7 : 2.2}" fill="#fff" fill-opacity="${big ? 0.95 : 0.28}"/>`;
    }
  } else if (kind === 2) {
    // 사선 띠
    const n = 5 + Math.floor(r() * 4);
    for (let i = 0; i < n; i++) {
      const x = -80 + i * (W / n) * 1.4, w = 14 + r() * 40;
      art += `<rect x="${x.toFixed(0)}" y="-60" width="${w.toFixed(0)}" height="${H + 120}" fill="#fff" fill-opacity="${(0.06 + r() * 0.16).toFixed(2)}" transform="rotate(-24 ${W / 2} ${H / 2})"/>`;
    }
    art += `<circle cx="${(W * 0.7).toFixed(0)}" cy="${(H * 0.3).toFixed(0)}" r="${(36 + r() * 30).toFixed(0)}" fill="#fff" fill-opacity=".88"/>`;
  } else {
    // 궤도
    const cx = W / 2, cy = H * 0.4;
    for (let i = 0; i < 4; i++) {
      const rx = 50 + i * 34, ry = 18 + i * 14, rot = r() * 180;
      art += `<ellipse cx="${cx}" cy="${cy.toFixed(0)}" rx="${rx}" ry="${ry}" fill="none" stroke="#fff" stroke-opacity=".35" stroke-width="1.5" transform="rotate(${rot.toFixed(0)} ${cx} ${cy.toFixed(0)})"/>`;
      const a = r() * Math.PI * 2;
      art += `<circle cx="${(cx + rx * Math.cos(a)).toFixed(0)}" cy="${(cy + ry * Math.sin(a)).toFixed(0)}" r="${4 + i}" fill="#fff" fill-opacity=".9" transform="rotate(${rot.toFixed(0)} ${cx} ${cy.toFixed(0)})"/>`;
    }
    art += `<circle cx="${cx}" cy="${cy.toFixed(0)}" r="22" fill="#fff" fill-opacity=".95"/>`;
  }
  const big = label ? `<text x="24" y="${H - 28}" font-family="Pretendard Variable, Pretendard, system-ui, sans-serif" font-size="${label.length > 4 ? 56 : 72}" font-weight="800" fill="#fff" letter-spacing="-2">${esc(label)}</text>` : "";
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid slice" aria-hidden="true">`
    + `<defs><linearGradient id="${gid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${c1}"/><stop offset="1" stop-color="${c2}"/></linearGradient></defs>`
    + `<rect width="${W}" height="${H}" fill="url(#${gid})"/>${art}`
    + (code ? `<rect x="18" y="18" rx="9" width="${code.length * 11 + 22}" height="26" fill="#000" fill-opacity=".22"/>`
      + `<text x="29" y="36" font-family="JetBrains Mono, ui-monospace, monospace" font-size="13" font-weight="700" fill="#fff">${esc(code)}</text>` : "")
    + big + `</svg>`;
}
