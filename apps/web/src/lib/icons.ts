// 사이트 공통 아이콘(SVG 문자열). 선 굵기 2, 둥근 끝.
const svg = (d: string, w = 2) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;

export const ICONS = {
  home: svg('<path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>'),
  map: svg('<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="M7.5 8l3.5 7.6M16.5 8 13 15.6M8.5 6h7"/>'),
  tracks: svg('<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/>'),
  progress: svg('<path d="M4 20V11M10 20V4M16 20v-6M22 20H2"/>'),
  search: svg('<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>'),
  theme: svg('<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"/>'),
  close: svg('<path d="M18 6 6 18M6 6l12 12"/>', 2.4),
  back: svg('<path d="M15 6l-6 6 6 6"/>', 2.4),
  arrow: svg('<path d="M5 12h14M13 6l6 6-6 6"/>', 2.4),
  /* 퀘스트 노드 */
  goal: svg('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/>', 2.4),
  read: svg('<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5zM4 20.5A2.5 2.5 0 0 0 6.5 23H20v-5"/>', 2.4),
  widget: svg('<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>', 2.4),
  code: svg('<path d="m8 8-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14"/>', 2.4),
  quiz: svg('<circle cx="12" cy="12" r="9"/><path d="M9.2 9.3a2.9 2.9 0 0 1 5.6 1c0 2-2.8 2.6-2.8 4.2M12 17.6h.01"/>', 2.4),
  lab: svg('<rect x="2.5" y="4" width="19" height="16" rx="3"/><path d="m7 10 3 2.5L7 15M12.5 15H17"/>', 2.4),
  pitfall: svg('<path d="M12 3 2 20h20zM12 10v4M12 17h.01"/>', 2.4),
  ref: svg('<path d="M6 3h9l3 3v15H6zM9 9h6M9 13h6M9 17h4"/>', 2.4),
  boss: svg('<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0zM7 6H4v1a3 3 0 0 0 3 3M17 6h3v1a3 3 0 0 1-3 3"/>', 2.4),
  done: svg('<path d="m5 12.5 4.5 4.5L19 7.5"/>', 2.8),
  /* 실습실 */
  guide: svg('<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5zM4 20.5A2.5 2.5 0 0 0 6.5 23H20v-5"/>'),
  editor: svg('<path d="m8 8-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14"/>'),
  terminal: svg('<rect x="2.5" y="4" width="19" height="16" rx="3"/><path d="m7 10 3 2.5L7 15M12.5 15H17"/>'),
  check: svg('<path d="m5 12.5 4.5 4.5L19 7.5"/>', 2.4),
  folder: svg('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'),
};
export type IconName = keyof typeof ICONS;
