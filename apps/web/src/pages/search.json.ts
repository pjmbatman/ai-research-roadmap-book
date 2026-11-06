// ⌘K 검색용 색인. 장·부·트랙의 제목과 영어 제목, 절 제목을 담는다.
import { loadBook, headings, FIXED, str } from "../lib/book";

export function GET() {
  const { parts, config } = loadBook();
  const items: { t: string; n: string; e: string; u: string; k: string; h: string }[] = [];
  for (const t of config.tracks as any[]) items.push({ t: t.name, n: t.code, e: t.name_en, u: `tracks/${t.key}/`, k: "트랙", h: t.question });
  for (const p of parts) {
    items.push({ t: str(p.meta.title), n: p.label === "부록" ? "부록" : `${p.label}부`, e: str(p.meta.title_en), u: p.url, k: "부", h: str(p.meta.summary) });
    for (const c of p.chapters) {
      const hs = headings(c.body).filter((h) => !FIXED.includes(h)).join(" · ");
      items.push({ t: str(c.meta.title), n: c.label, e: str(c.meta.title_en), u: c.url, k: "장", h: hs });
    }
  }
  return new Response(JSON.stringify(items), { headers: { "Content-Type": "application/json" } });
}
