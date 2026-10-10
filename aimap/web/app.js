"use strict";
/* AI 산업 지도 — a phone-first reading guide drawn from data.json (aimap/build.py).
   Routes: #/ route map · #/s/<stop>[/<zone>] · #/s/<stop>/view/<panel> (full cutaway)
           #/c/<ticker> · #/find · #/words[/<term>] · #/axes · #/quiz
   Rules: docs/desk_system.md §8.3. */

const V = document.documentElement.dataset.v || "";
const NS = "http://www.w3.org/2000/svg";
const $ = (sel, root = document) => root.querySelector(sel);

function dress(el, attrs, kids) {
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  add(el, kids);
  return el;
}
function add(el, kids) {
  for (const k of kids.flat(Infinity)) if (k != null && k !== false) el.append(k instanceof Node ? k : String(k));
}
const h = (tag, attrs, ...kids) => dress(document.createElement(tag), attrs, kids);
const s = (tag, attrs, ...kids) => dress(document.createElementNS(NS, tag), attrs, kids);
const mount = (el, ...kids) => { el.replaceChildren(); add(el, kids); };

const store = {
  get(k, d) { try { const v = localStorage.getItem("aimap-" + k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } },
  set(k, v) { try { localStorage.setItem("aimap-" + k, JSON.stringify(v)); } catch (_) { /* private mode: nothing kept */ } },
};

let D = null;   // data.json
let I = null;   // indexes over D

/* ------------------------------------------------------------------ data */
function index(d) {
  const stopById = {}, zoneById = {}, spots = {}, links = {}, axesOf = {}, main = {};
  for (const st of d.stops) {
    stopById[st.id] = st;
    for (const z of st.zones) {
      zoneById[z.id] = { st, z };
      for (const m of z.members) (spots[m.t] ||= []).push({ st, z, m });
    }
  }
  for (const [t, list] of Object.entries(spots)) main[t] = list.find((x) => x.m.main) || list[0];
  for (const ln of d.links) (links[ln.t] ||= []).push(ln);
  for (const [t, list] of Object.entries(links)) {
    list.sort((a, b) => b.date.localeCompare(a.date));
    axesOf[t] = new Set(list.map((l) => l.axis));
  }
  return { stopById, zoneById, spots, links, axesOf, main, terms: termIndex(d.glossary) };
}

const MARKET = { US: "미국", KR: "한국", JP: "일본", TW: "대만", HK: "홍콩", CN: "중국", EU: "유럽", IN: "인도", private: "비상장" };
const AX = { openai: { ko: "OpenAI", cls: "oa", prod: "ChatGPT" }, anthropic: { ko: "Anthropic", cls: "an", prod: "Claude" } };
const INK = { power: "#f0b429", cold: "#3b8fe0", hot: "#e8603c", fiber: "#25bfe0", light: "#c690ff" };  // colours baked into the renders
const co = (t) => D.companies[t] || { name: t, ko: t, market: "US" };
const koName = (t) => co(t).ko || co(t).name || t;
const code = (t) => (co(t).market === "private" ? "비상장" : t);
const codeTag = (t) => (code(t) === koName(t) ? null : h("span", { class: "tk" }, code(t)));   // "ASML ASML" → "ASML"
const axesOf = (t) => I.axesOf[t] || new Set();
const usd = (bn) => (bn >= 1000 ? `${+(bn / 1000).toFixed(bn % 1000 ? 1 : 0)}조 달러` : `${Math.round(bn * 10).toLocaleString("ko-KR")}억 달러`);
const leadFirst = (z) => [...z.members].sort((a, b) => (b.t === z.lead) - (a.t === z.lead));
/* 저평가 표시(desk/value.py, docs/desk_system.md §8.4). 빌드가 오프라인이면 D.value가 없고 표시도 없다. */
const uvZone = (zid) => (D.value && D.value.zones[zid]) || null;
const uvCo = (t) => (D.value && D.value.companies[t]) || null;
const pct = (x, d = 1) => (x == null ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(d)}%`);
const pp = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(1)}`;   // 이미 %p 단위인 검증 숫자
const CYCLE_NOTE = "메모리·저장은 경기순환 업종입니다. 이익이 정점일 때 PER이 가장 낮아 보이므로, 전망이 오르는데 주가가 빠지는 것은 업황이 꺾이는 신호일 수도 있습니다.";

/* ------------------------------------------------------------- glossary */
const PARTICLE = "(?:으로는|으로도|으로|에서|에게|까지|부터|보다|처럼|이나|이며|이고|이자|과의|와의|에도|에는|로는|로도|은|는|이|가|을|를|와|과|도|로|의|에|만|나)";
const HANGUL = /[가-힣]/;
function termIndex(glossary) {
  const alts = [];
  for (const g of glossary) for (const m of g.match) alts.push({ m, g });
  alts.sort((a, b) => b.m.length - a.m.length);
  const esc = (x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const edge = (m) => {
    const a = m[0], z = m[m.length - 1];
    const pre = HANGUL.test(a) ? "(?<![가-힣A-Za-z0-9])" : /\d/.test(a) ? "(?<![0-9.])" : "(?<![A-Za-z])";
    const post = HANGUL.test(z) ? `(?=${PARTICLE}?(?![가-힣]))` : /\d/.test(z) ? "(?![0-9])" : "(?![A-Za-z])";
    return pre + esc(m) + post;
  };
  let re = null;
  try { re = new RegExp(alts.map((x) => edge(x.m)).join("|"), "g"); } catch (_) { /* no lookbehind (iOS < 16.4): plain text */ }
  return { re, of: new Map(alts.map((x) => [x.m, x.g])) };
}
/* Text with glossary words turned into buttons — each word only the first time on a page. */
let seenTerms = new Set();
function rich(text) {
  if (!text) return null;
  if (!I.terms.re) return text;
  const out = [];
  let last = 0;
  for (const m of text.matchAll(I.terms.re)) {
    const g = I.terms.of.get(m[0]);
    if (!g || seenTerms.has(g.term)) continue;
    seenTerms.add(g.term);
    out.push(text.slice(last, m.index),
      h("button", { type: "button", class: "term", "data-term": g.term, onclick: () => openTerm(g) }, m[0]));
    last = m.index + m[0].length;
  }
  out.push(text.slice(last));
  return out;
}

/* ------------------------------------------------------------- icons */
const ICONS = {
  power: [["path", { d: "M13 2 4.5 13.5H11L10 22l8.5-11.5H12z" }]],
  inputs: [["path", { d: "M9 3h6M10 3v6.2L4.6 18.4A2 2 0 0 0 6.3 21.5h11.4a2 2 0 0 0 1.7-3.1L14 9.2V3M7.5 15h9" }]],
  tools: [["path", { d: "M14.7 6.3a4 4 0 0 0-5.3 5.3L3 18l3 3 6.4-6.4a4 4 0 0 0 5.3-5.3l-2.6 2.6-2.3-.6-.6-2.3z" }]],
  fab: [["path", { d: "M3 21V10l5 3v-3l5 3v-3l5 3V4h3v17zM7 17h2M12 17h2M17 17h1" }]],
  packaging: [["path", { d: "m3 7.5 9-4.5 9 4.5-9 4.5zM3 7.5v9l9 4.5 9-4.5v-9M12 12v9" }]],
  chips: [["rect", { x: 7, y: 7, width: 10, height: 10, rx: 1.5 }], ["path", { d: "M10 2v3M14 2v3M10 19v3M14 19v3M2 10h3M2 14h3M19 10h3M19 14h3M10 10h4v4h-4z" }]],
  parts: [["rect", { x: 3, y: 4, width: 18, height: 16, rx: 2 }], ["path", { d: "M7 8h3v3H7zM14 13h3v3h-3zM14 8.5h3M7 15h3M10 9.5h4M10.5 14.5H14" }]],
  assembly: [["rect", { x: 5, y: 2.5, width: 14, height: 19, rx: 1.5 }], ["path", { d: "M8 6.5h8M8 10.5h8M8 14.5h8M8 18.5h2" }]],
  network: [["circle", { cx: 12, cy: 5, r: 2.3 }], ["circle", { cx: 5, cy: 18.5, r: 2.3 }], ["circle", { cx: 19, cy: 18.5, r: 2.3 }], ["path", { d: "m10.9 7.1-4.8 9.3M13.1 7.1l4.8 9.3M7.3 18.5h9.4" }]],
  storage: [["ellipse", { cx: 12, cy: 5.5, rx: 7.5, ry: 2.8 }], ["path", { d: "M4.5 5.5v13c0 1.6 3.4 2.8 7.5 2.8s7.5-1.2 7.5-2.8v-13M4.5 12c0 1.6 3.4 2.8 7.5 2.8s7.5-1.2 7.5-2.8" }]],
  datacenter: [["path", { d: "M3 21h18M5 21V7.5L12 3l7 4.5V21M9 9.5h2M13 9.5h2M9 13.5h2M13 13.5h2M10 21v-4h4v4" }]],
  cloud: [["path", { d: "M7 19h10.2a4.3 4.3 0 0 0 .6-8.6A6.2 6.2 0 0 0 5.9 9.7 4.7 4.7 0 0 0 7 19z" }]],
  labs: [["path", { d: "m11 3 1.9 5.6L18.5 10.5l-5.6 1.9L11 18l-1.9-5.6L3.5 10.5l5.6-1.9zM18.5 15l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z" }]],
  apps: [["rect", { x: 6.5, y: 2, width: 11, height: 20, rx: 2.4 }], ["path", { d: "M10.5 18.5h3" }]],
  data: [["ellipse", { cx: 12, cy: 6, rx: 7, ry: 2.6 }], ["path", { d: "M5 6v12c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6V6M5 12c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6" }]],
  enterprise: [["rect", { x: 3, y: 7, width: 18, height: 13, rx: 2 }], ["path", { d: "M9 7V5h6v2M3 12.5h18" }]],
  services: [["circle", { cx: 9, cy: 8, r: 3 }], ["circle", { cx: 16.5, cy: 9, r: 2.4 }], ["path", { d: "M3.5 19.5c.6-3.2 2.8-5 5.5-5s4.9 1.8 5.5 5M14.5 14.6c2.7-.5 5.3 1 6 4.4" }]],
  security: [["path", { d: "M12 2.5 4.5 5.5v6c0 4.8 3.2 8.4 7.5 10 4.3-1.6 7.5-5.2 7.5-10v-6zM9 12l2.2 2.2L15.5 10" }]],
  devtools: [["path", { d: "m8.5 7-5 5 5 5M15.5 7l5 5-5 5M13.5 4.5l-3 15" }]],
  consumer: [["circle", { cx: 10.5, cy: 10.5, r: 6.5 }], ["path", { d: "m15.5 15.5 5 5" }]],
  device: [["rect", { x: 3, y: 4, width: 13, height: 10, rx: 1.5 }], ["path", { d: "M1.5 17h16" }], ["rect", { x: 17.5, y: 8, width: 5, height: 11, rx: 1.2 }]],
  physical: [["rect", { x: 5, y: 7, width: 14, height: 10, rx: 3 }], ["path", { d: "M12 7V3.5M9.5 11.5v1M14.5 11.5v1M8 17v3.5M16 17v3.5M2.5 11v3M21.5 11v3" }]],
};
const LEG_ICON = { "기초": "power", "칩 만들기": "chips", "서버 만들기": "assembly", "AI 공장": "datacenter", "완제품": "labs" };
function icon(name, cls = "ic") {
  return s("svg", { class: cls, viewBox: "0 0 24 24", "aria-hidden": "true" },
    (ICONS[name] || ICONS.labs).map(([tag, at]) => s(tag, at)));
}

/* ------------------------------------------------------------- shell */
function setBar({ back = null, title = "", progress = false } = {}) {
  const theme = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  mount($("#bar"),
    back ? h("a", { class: "bar-back", href: back.href, "aria-label": back.label }, h("span", { "aria-hidden": "true" }, "←"), h("span", { class: "lbl" }, back.label))
      : h("a", { class: "bar-brand", href: "#/" }, icon("labs", "ic brand-ic"), "AI 산업 지도"),
    h("span", { class: "bar-title" }, title),
    h("a", { class: "bar-btn", href: "#/find", "aria-label": "회사·용어 찾기" },
      s("svg", { viewBox: "0 0 24 24", class: "ic", "aria-hidden": "true" }, s("circle", { cx: 10.5, cy: 10.5, r: 6.5 }), s("path", { d: "m15.5 15.5 5 5" }))),
    h("button", { type: "button", class: "bar-btn", "aria-label": theme === "dark" ? "밝은 화면으로" : "어두운 화면으로", onclick: toggleTheme },
      s("svg", { viewBox: "0 0 24 24", class: "ic", "aria-hidden": "true" },
        theme === "dark" ? [s("circle", { cx: 12, cy: 12, r: 4.5 }), s("path", { d: "M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" })]
          : s("path", { d: "M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" }))));
  document.body.classList.toggle("reading", progress);
  readProgress();
}
function toggleTheme() {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  store.set("theme", next);
  const bar = $("#bar .bar-title");
  setBar({ back: barState.back, title: bar ? bar.textContent : "", progress: barState.progress });
}
let barState = {};
function bar(opts) { barState = opts; setBar(opts); }

function readProgress() {
  const el = $("#readbar i");
  if (!el) return;
  const max = document.documentElement.scrollHeight - innerHeight;
  el.style.transform = `scaleX(${max > 0 ? Math.min(1, scrollY / max) : 0})`;
}

/* ------------------------------------------------------------- router */
const scrollMem = new Map();
let pushing = false, shownKey = null, curHash = null, depth = 0;
history.scrollRestoration = "manual";
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[href^='#/']");
  if (a && e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey) pushing = true;
});
addEventListener("scroll", () => {
  if (curHash != null) scrollMem.set(curHash, scrollY);
  readProgress();
}, { passive: true });
addEventListener("hashchange", route);
function go(hash) { pushing = true; location.hash = hash; }
function back(fallback) { if (depth > 0) history.back(); else location.replace(fallback); }

function route() {
  const push = pushing;
  pushing = false;
  depth = push ? depth + 1 : Math.max(0, depth - 1);
  const hash = location.hash && location.hash !== "#" ? location.hash : "#/";
  const [a, b, c, d] = hash.replace(/^#\/?/, "").split("/").map((x) => decodeURIComponent(x || ""));
  closeSheet();
  let key = "home", render = viewHome, after = null;
  if (a === "s" && I.stopById[b]) {
    const st = I.stopById[b];
    key = "s/" + b;
    render = () => viewStop(st);
    after = c === "view" && D.panels[d] ? () => openViewer(st, d)
      : (same) => { closeViewer(); if (c && I.zoneById[c] && (push || !scrollMem.has(hash))) toZone(c, same); };
  } else if (a === "c" && b && (D.companies[b] || I.spots[b] || I.links[b])) {
    key = "c/" + b; render = () => viewCompany(b);
  } else if (a === "find") { key = "find"; render = viewFind; }
  else if (a === "words") { key = "words"; render = viewWords; after = () => b && toId("w-" + slug(b)); }
  else if (a === "axes") { key = "axes"; render = viewAxes; }
  else if (a === "quiz") { key = "quiz"; render = viewQuiz; }
  else if (a === "value") { key = "value"; render = viewValue; }
  const same = key === shownKey;
  if (!same) {
    closeViewer();
    pinWatch.disconnect();
    seenTerms = new Set();
    render();
    shownKey = key;
    scrollTo(0, push ? 0 : scrollMem.get(hash) || 0);
  }
  curHash = hash;
  if (after) after(same);
  readProgress();
}
const slug = (x) => encodeURIComponent(x).replace(/%/g, "");
function toId(id, smooth = false) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: smooth && !reduced() ? "smooth" : "auto", block: "start" });
}
function toZone(zid, smooth) {
  const el = document.getElementById("z-" + zid);
  if (!el) return;
  el.scrollIntoView({ behavior: smooth && !reduced() ? "smooth" : "auto", block: "start" });
  el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
}
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ------------------------------------------------------------- home */
function viewHome() {
  bar({});
  const visited = new Set(store.get("visited", []));
  const last = I.stopById[store.get("last")] || null;
  const legs = [...new Set(D.stops.map((st) => st.leg))];
  mount($("#app"),
    h("section", { class: "cover" },
      h("div", { class: "wrap" },
        h("p", { class: "kicker" }, "AI 산업 지도 · 쉬운 해설"),
        h("h1", { class: "cover-title" }, h("span", null, "전기에서"), h("span", null, "ChatGPT까지")),
        h("p", { class: "lede" }, D.intro.lede),
        h("ul", { class: "stats" },
          [[D.counts.stops, "곳의 일터"], [D.counts.zones, "개 분야"], [D.counts.companies, "개 회사"]].map(([n, l]) => h("li", null, h("b", null, n), l))),
        h("div", { class: "cta" },
          h("a", { class: "btn primary", href: "#/s/" + D.stops[0].id }, "1번 전력부터 읽기", h("span", { "aria-hidden": "true" }, " →")),
          last && last.no > 1 ? h("a", { class: "btn", href: "#/s/" + last.id }, `이어 읽기 · ${last.no}. ${last.name}`) : null),
        h("p", { class: "asof" }, `지도 기준일 ${D.as_of} · 방문 ${visited.size}/${D.stops.length}`)),
      h("div", { class: "cover-art", "aria-hidden": "true" }, coverArt())),
    h("section", { class: "wrap sec-block" },
      h("h2", { class: "sec" }, "1분 요약"),
      h("p", { class: "big-line" }, D.intro.big),
      h("ol", { class: "legs" }, legs.map((leg, i) => {
        const stops = D.stops.filter((st) => st.leg === leg);
        return h("li", null, h("a", { href: "#/", onclick: (e) => { e.preventDefault(); toId("leg-" + i, true); } },
          h("span", { class: "leg-ic" }, icon(LEG_ICON[leg] || "labs")),
          h("b", null, leg), h("span", { class: "leg-sub" }, D.legs[leg].title),
          h("span", { class: "leg-no" }, `${stops[0].no}–${stops[stops.length - 1].no}`)));
      }))),
    h("section", { class: "wrap sec-block" },
      h("h2", { class: "sec" }, "읽는 법"),
      h("ol", { class: "how" }, D.intro.how.map((t) => h("li", null, t)))),
    uvHome(),
    h("section", { class: "wrap sec-block" },
      h("h2", { class: "sec" }, "노선도", h("small", null, "번호 = 만드는 순서")),
      routeList(visited, legs)),
    h("section", { class: "wrap sec-block" },
      h("h2", { class: "sec" }, "다른 방법으로 익히기"),
      h("div", { class: "tiles" },
        tile("#/find", "회사 찾기", `${D.counts.companies}개 회사를 이름·티커로 찾고, 어느 일터에서 무슨 일을 하는지 봅니다.`),
        tile("#/words", "용어 사전", `HBM, CoWoS, EUV처럼 자주 나오는 말 ${D.glossary.length}개를 쉬운 말로 풀었습니다.`),
        tile("#/axes", "OpenAI·Anthropic의 돈 흐름", "두 회사가 누구와 어떤 계약을 맺었는지 한눈에 봅니다."),
        tile("#/quiz", "퀴즈로 익히기", "이 회사는 어느 일터에서 일할까? 맞히며 외웁니다."),
        D.value ? tile("#/value", "저평가 표시는 이렇게 정했습니다", "PER 수준 대신 '전망 대비 주가'의 30일 변화를 분야끼리 비교합니다. 검증과 한계도 함께.") : null)),
    foot());
}
function tile(href, title, text) {
  return h("a", { class: "tile", href }, h("b", null, title), h("span", null, text), h("i", { "aria-hidden": "true" }, "→"));
}
function coverArt() {
  const hero = D.stops.find((st) => st.id === "datacenter") || D.stops.find((st) => st.heroes && st.heroes.length);
  const pic = hero && hero.heroes && hero.heroes[0];
  return pic ? h("img", { src: D.panels[pic.panel].img, alt: "", width: D.panels[pic.panel].w, height: D.panels[pic.panel].h, decoding: "async" }) : null;
}
function routeList(visited, legs) {
  const ol = h("ol", { class: "route" });
  let leg = null;
  for (const st of D.stops) {
    if (st.leg !== leg) {
      leg = st.leg;
      ol.append(h("li", { class: "leg-head", id: "leg-" + legs.indexOf(leg) },
        h("span", { class: "leg-name" }, leg), h("span", { class: "leg-title" }, D.legs[leg].title), h("p", null, D.legs[leg].text)));
    }
    const nCo = new Set(st.zones.flatMap((z) => z.members.map((m) => m.t))).size;
    const nUV = st.zones.filter((z) => uvZone(z.id) && uvZone(z.id).flag).length;
    ol.append(h("li", { class: "station" + (visited.has(st.id) ? " seen" : "") + (st.id === "labs" ? " product" : ""), style: `--i:${st.no}` },
      h("a", { href: "#/s/" + st.id },
        h("span", { class: "dot" }, st.no),
        h("span", { class: "st-body" },
          h("b", { class: "st-name" }, st.name, visited.has(st.id) ? h("i", { class: "seen-mark", "aria-label": "읽음" }, "✓") : null),
          h("span", { class: "st-site" }, st.site),
          h("span", { class: "st-easy" }, st.easy),
          h("span", { class: "st-meta" }, h("span", null, `만드는 것 · ${st.makes}`), h("span", null, `분야 ${st.zones.length} · 회사 ${nCo}`),
            nUV ? h("span", { class: "uv-tag sm" }, `저평가 ${nUV}`) : null)),
        st.thumb ? h("img", { class: "st-thumb", src: st.thumb, alt: "", loading: "lazy", decoding: "async", width: 360, height: 270 })
          : h("span", { class: "st-thumb is-icon" }, icon(st.id)))));
  }
  ol.append(h("li", { class: "station end" }, h("span", { class: "dot" }, "★"), h("span", { class: "st-body" }, h("b", { class: "st-name" }, "고객 — 우리"), h("span", { class: "st-easy" }, "ChatGPT·Claude 같은 서비스로 답을 받습니다."))));
  return ol;
}
function foot() {
  return h("footer", { class: "foot wrap" },
    h("p", null, `지도 기준일 ${D.as_of} · 사이트 빌드 ${D.built}`, D.value ? ` · 저평가 ${D.value.price_date} 종가 기준` : ""),
    h("p", null, "회사 역할과 숫자는 원문 출처를 함께 적었습니다. 그림은 이해를 돕는 일러스트로 실제 크기·개수·배치와 다르며 특정 회사 제품이 아닙니다."),
    h("p", null, "투자 권유가 아니라 산업 구조를 이해하기 위한 자료입니다. ★ 대표 기업은 점유율·매출·대표성으로 고른 것이며 모두 점유율 1위라는 뜻은 아닙니다."));
}

/* ------------------------------------------------------------- stop */
function markVisited(id) {
  const v = new Set(store.get("visited", []));
  v.add(id);
  store.set("visited", [...v]);
  store.set("last", id);
}
function viewStop(st) {
  markVisited(st.id);
  const at = D.stops.indexOf(st), prev = D.stops[at - 1], next = D.stops[at + 1];
  bar({ back: { href: "#/", label: "노선도" }, title: `${st.no} / ${D.stops.length}`, progress: true });
  const hop = (id) => { const x = I.stopById[id]; return h("a", { class: "hop", href: "#/s/" + id }, `${x.no}. ${x.name}`); };
  mount($("#app"),
    h("article", { class: "stop" },
      h("header", { class: "wrap stop-head" },
        h("p", { class: "kicker" }, `${st.leg} · ${st.no}번째 일터`),
        h("h1", { class: "stop-title" }, st.name),
        h("p", { class: "stop-site" }, st.site),
        h("p", { class: "lede" }, rich(st.easy))),
      h("div", { class: "wrap" },
        h("aside", { class: "analogy" }, h("span", { class: "tag-label" }, "비유하면"), h("p", null, rich(st.analogy))),
        h("ol", { class: "io" },
          h("li", null, h("span", { class: "io-k" }, "받는 것"), h("span", { class: "io-v" }, st.gets),
            h("span", { class: "hops" }, (st.from || []).length ? st.from.map(hop) : h("span", { class: "hop start" }, "출발점"))),
          h("li", { class: "make" }, h("span", { class: "io-k" }, "여기서 만드는 것"), h("span", { class: "io-v" }, st.makes)),
          h("li", null, h("span", { class: "io-k" }, "넘기는 것"), h("span", { class: "io-v" }, st.gives),
            h("span", { class: "hops" }, (st.to || []).length ? st.to.map(hop) : h("span", { class: "hop start" }, "고객(우리)"))))),
      pictures(st),
      h("div", { class: "wrap" },
        h("section", { class: "why" }, h("h2", { class: "sec" }, "왜 중요할까"), h("p", null, rich(st.need))),
        h("section", { class: "why money" }, h("h2", { class: "sec" }, "돈이 들어오는 길"), h("p", null, rich(st.money))),
        h("nav", { class: "zone-index", "aria-label": `${st.name}의 분야` },
          h("h2", { class: "sec" }, "이곳의 분야", h("small", null, `${st.zones.length}개`)),
          h("ul", null, st.zones.map((z) => h("li", null,
            h("a", { href: `#/s/${st.id}/${z.id}` }, h("span", { class: "zn" }, z.n), z.name,
              h("span", { class: "flags" },
                z.bottleneck ? h("span", { class: "bn-dot", title: "병목" }, "병목") : null,
                uvZone(z.id) && uvZone(z.id).flag ? h("span", { class: "uv-tag sm" }, "저평가") : null))))))),
      st.zones.map((z) => zoneSection(st, z)),
      h("nav", { class: "wrap next-nav" },
        prev ? h("a", { class: "btn", href: "#/s/" + prev.id }, `← ${prev.no}. ${prev.name}`) : h("span"),
        next ? h("a", { class: "btn primary", href: "#/s/" + next.id }, `다음 일터 · ${next.no}. ${next.name} →`)
          : h("a", { class: "btn primary", href: "#/quiz" }, "다 읽었습니다 · 퀴즈로 확인하기 →")),
      foot()));
}

function pictures(st) {
  if (st.heroes && st.heroes.length) {
    const many = st.heroes.length > 1;
    return h("section", { class: "plates" + (many ? " many" : "") },
      h("div", { class: "plate-row" }, st.heroes.map((pic) => {
        const p = D.panels[pic.panel];
        const box = h("div", { class: "plate-img" },
          h("img", { src: pic.img, alt: `${p.title} — ${st.name} 분야를 밝힌 그림`, width: pic.w, height: pic.h, decoding: "async" }),
          s("svg", { class: "pin-lines", "aria-hidden": "true" }),
          pic.dots.map((dt) => {
            const z = I.zoneById[dt.zone].z;
            return h("a", { class: "pin", href: `#/s/${st.id}/${dt.zone}`, "data-x": dt.x, "data-y": dt.y,
                            style: `left:${dt.x * 100}%;top:${dt.y * 100}%`, "aria-label": `${z.n} ${z.name}` }, z.n);
          }));
        pinWatch.observe(box);
        return h("figure", { class: "plate" },
          box,
          h("figcaption", null,
            h("b", null, p.title), " ", p.caption,
            p.legend.length ? h("span", { class: "legend" }, p.legend.map(([k, label]) => h("span", null, h("i", { style: `background:${INK[k]}` }), label))) : null,
            h("a", { class: "zoom", href: `#/s/${st.id}/view/${pic.panel}` }, "크게 보기 · 이름표")));
      })),
      many ? h("p", { class: "swipe-hint wrap" }, `그림 ${st.heroes.length}장 · 옆으로 넘겨 보세요`) : null);
  }
  const dg = DIAGRAMS[st.id];
  return dg ? h("section", { class: "wrap" }, h("figure", { class: "plate diagram" }, dg(st))) : null;
}

/* Numbered pins on a lit picture: each starts on its zone's anchor; a pin that would cover an
   earlier one moves to the nearest free spot around its anchor and keeps a short leader line. */
const pinWatch = new ResizeObserver((entries) => { for (const e of entries) layoutPins(e.target); });
function layoutPins(box) {
  const W = box.clientWidth, H = box.clientHeight, lines = box.querySelector(".pin-lines");
  if (!W || !H || !lines) return;
  lines.setAttribute("viewBox", `0 0 ${W} ${H}`);
  lines.replaceChildren();
  const placed = [], gap = 3;
  const free = (x, y, w, hh) => x - w / 2 >= 2 && x + w / 2 <= W - 2 && y - hh / 2 >= 2 && y + hh / 2 <= H - 2
    && placed.every((r) => Math.abs(x - r.x) >= (w + r.w) / 2 + gap || Math.abs(y - r.y) >= (hh + r.h) / 2 + gap);
  for (const pin of box.querySelectorAll(".pin")) {
    const ax = +pin.dataset.x * W, ay = +pin.dataset.y * H, w = pin.offsetWidth, hh = pin.offsetHeight;
    let at = [ax, ay];
    if (!free(ax, ay, w, hh)) {
      const cands = [];
      for (let k = 1; k <= 4; k++) for (const [dx, dy] of [[0, -1], [0, 1], [1, 0], [-1, 0], [1, -1], [-1, -1], [1, 1], [-1, 1]])
        cands.push([ax + dx * k * (w * 0.55 + gap), ay + dy * k * (hh + gap)]);
      cands.sort((a, b) => Math.hypot(a[0] - ax, a[1] - ay) - Math.hypot(b[0] - ax, b[1] - ay));
      at = cands.find(([x, y]) => free(x, y, w, hh)) || at;
    }
    placed.push({ x: at[0], y: at[1], w, h: hh });
    pin.style.left = `${at[0]}px`;
    pin.style.top = `${at[1]}px`;
    if (at[0] !== ax || at[1] !== ay) lines.append(s("line", { x1: ax, y1: ay, x2: at[0], y2: at[1] }));
    lines.append(s("circle", { cx: ax, cy: ay, r: 3 }));
  }
}

function zoneSection(st, z) {
  const members = leadFirst(z), more = members.length > 5;
  const list = h("ul", { class: "cos" + (more ? " folded" : "") }, members.map((m, i) => coRow(m, z, i >= 5)));
  const lead = z.lead && z.members.find((m) => m.t === z.lead);
  const leadSrc = z.lead_src || (lead && lead.src);
  const alone = lead && z.members.length === 1;   // the lead card already is the whole list
  return h("section", { class: "zone wrap", id: "z-" + z.id },
    h("header", { class: "zone-head" },
      h("span", { class: "zn" }, z.n),
      h("h2", null, z.name),
      z.bottleneck ? h("span", { class: "bn-tag" }, "병목") : null,
      uvZone(z.id) && uvZone(z.id).flag ? h("span", { class: "uv-tag" }, "저평가") : null),
    z.crops && z.crops.length ? h("div", { class: "crops" + (z.crops.length > 1 ? " many" : "") }, z.crops.map((c) =>
      h("figure", { class: "crop" },
        h("a", { href: `#/s/${st.id}/view/${c.panel}`, "aria-label": `${D.panels[c.panel].title} 크게 보기` },
          h("img", { src: c.img, alt: `${z.name} 부분을 밝힌 그림 (${D.panels[c.panel].title})`, width: c.w, height: c.h, loading: "lazy", decoding: "async" })),
        h("figcaption", null, D.panels[c.panel].title, z.crops.length > 1 ? h("span", null, ` · ${z.crops.indexOf(c) + 1}/${z.crops.length}`) : null)))) : null,
    z.special === "axes" ? axesPair() : null,
    h("p", { class: "easy" }, rich(z.easy)),
    z.need && z.special !== "axes" ? h("p", { class: "need" }, rich(z.need)) : null,
    z.bottleneck ? h("div", { class: "bn-box" }, h("b", null, "병목인 이유"), h("p", null, rich(z.bottleneck), z.src ? src(z.src) : null)) : null,
    uvBox(z),
    lead ? h("div", { class: "lead-card" },
      h("span", { class: "tag-label" }, alone ? "★ 대표 기업 · 이 분야의 유일한 회사" : "★ 대표 기업"),
      h("a", { class: "lead-co", href: "#/c/" + encodeURIComponent(z.lead) },
        h("b", null, koName(z.lead)), codeTag(z.lead), h("span", { class: "go", "aria-hidden": "true" }, "→")),
      alone && lead.role ? h("p", null, lead.role, lead.src || leadSrc ? src(lead.src || leadSrc) : null)
        : h("p", null, z.lead_why, leadSrc ? src(leadSrc) : null)) : null,
    z.members.length && !alone ? h("div", { class: "cos-wrap" },
      h("h3", { class: "cos-title" }, `회사 ${z.members.length}곳`, h("small", null, "누르면 회사 소개")),
      list,
      more ? h("button", { type: "button", class: "more", onclick: (e) => { list.classList.remove("folded"); e.currentTarget.remove(); } },
        `나머지 ${members.length - 5}곳 더 보기`) : null) : null,
    z.race ? h("details", { class: "race" }, h("summary", null, "경쟁 구도"), h("p", null, rich(z.race), z.src && !z.bottleneck ? src(z.src) : null)) : null);
}
const src = (href) => h("a", { class: "src", href, target: "_blank", rel: "noopener noreferrer" }, "출처");

function uvBox(z) {
  const V = D.value, v = uvZone(z.id);
  if (!V || !v) return null;
  const head = `최근 ${V.window.days}일 이익 전망 ${pct(v.R)}, 주가 ${pct(v.P)} → 선행 PER ${pct(v.V)}`;
  if (!v.flag) return h("p", { class: "uv-line" }, `전망 대비 주가 · ${head} (AI 전체 ${pct(V.universe.V)})`);
  return h("div", { class: "uv-box" },
    h("b", null, "저평가인 이유"),
    h("p", null, `${head}. 같은 기간 AI 전체(중앙값)의 선행 PER은 ${pct(V.universe.V)}였습니다. ${v.of}개 분야 중 ${v.rank}번째로 싸졌습니다(${v.n}곳 중앙값).`),
    v.cycle ? h("p", { class: "uv-warn" }, CYCLE_NOTE) : null,
    h("a", { class: "uv-more", href: "#/value" }, "기준과 검증 보기 →"));
}
function uvRows(rows) {
  return h("ul", { class: "uv-list" }, rows.map(({ st, z, v }) => h("li", { class: v.flag ? "on" : null },
    h("a", { href: `#/s/${st.id}/${z.id}` },
      h("span", { class: "zn" }, z.n),
      h("span", { class: "uv-name" }, h("b", null, z.name), h("small", null, `${st.no}. ${st.name} · ${v.n}곳${v.cycle ? " · 경기순환" : ""}`)),
      h("span", { class: "uv-num" }, `PER ${pct(v.V)}`, h("small", null, `전망 ${pct(v.R)} · 주가 ${pct(v.P)}`))))));
}
const uvZonesAll = () => D.stops.flatMap((st) => st.zones.map((z) => ({ st, z, v: uvZone(z.id) })))
  .filter((x) => x.v).sort((a, b) => a.v.rank - b.v.rank);
function uvHome() {
  const V = D.value;
  if (!V) return null;
  const flagged = uvZonesAll().filter((x) => x.v.flag);
  return h("section", { class: "wrap sec-block" },
    h("h2", { class: "sec" }, "지금 저평가 분야", h("small", null, `최근 ${V.window.days}일 · ${V.price_date} 종가`)),
    h("p", { class: "uv-intro" }, "이익 전망은 그대로이거나 올랐는데 주가가 덜 따라가 선행 PER이 내려간 분야입니다. 한 달 단위로 바뀝니다.",
      ` 같은 기간 AI 전체(중앙값)의 선행 PER은 ${pct(V.universe.V)}였습니다.`),
    flagged.length ? uvRows(flagged) : h("p", { class: "muted" }, "지금은 조건을 채운 분야가 없습니다."),
    h("a", { class: "uv-more", href: "#/value" }, "기준·검증·저평가 회사 보기 →"));
}
function uvCompany(t) {
  const V = D.value, c = uvCo(t);
  if (!V || !c) return null;
  const hit = c.zone && I.zoneById[c.zone], zname = hit ? `「${hit.z.name}」` : "", zv = c.zone ? uvZone(c.zone) : null;
  const verdict = c.skip === "loss" ? "적자라 PER로 판단하지 않습니다."
    : c.skip === "thin" ? "애널리스트가 3명 미만이라 판단하지 않습니다."
      : c.skip === "no_data" ? "이익 전망 자료가 없어 판단하지 않습니다."
        : c.flag ? `저평가로 표시합니다. 대표 분야 ${zname}가 저평가 분야이고, 이 회사도 전망은 오르고 PER은 내렸습니다.`
          : !zv ? `대표 분야 ${zname}는 잴 수 있는 회사가 3곳 미만이라 분야 판정이 없습니다.`
            : !zv.flag ? `대표 분야 ${zname}가 지금 저평가 분야가 아닙니다(${zv.of}개 중 싸진 순서 ${zv.rank}위).`
              : c.R <= 0 ? `대표 분야 ${zname}는 저평가지만 이 회사의 이익 전망은 오르지 않았습니다.`
                : `대표 분야 ${zname}는 저평가지만 이 회사의 선행 PER은 내려가지 않았습니다.`;
  const cell = (label, val) => h("div", null, h("span", null, label), h("b", null, val));
  return h("section", { class: "co-sec" },
    h("h2", { class: "sec" }, "전망 대비 주가", h("small", null, `최근 ${V.window.days}일 · ${V.price_date} 종가`),
      c.flag ? h("span", { class: "uv-tag" }, "저평가") : null),
    c.V != null ? h("div", { class: "uv-stats" },
      cell("이익 전망", pct(c.R)), cell("주가", pct(c.P)), cell("선행 PER", pct(c.V)), cell("AI 전체 PER", pct(V.universe.V))) : null,
    h("p", { class: "uv-verdict" }, verdict, " ", h("a", { href: "#/value" }, "기준 보기")),
    c.flag && zv && zv.cycle ? h("p", { class: "uv-warn" }, CYCLE_NOTE) : null);
}

function coRow(m, z, hidden) {
  const t = m.t, c = co(t), ax = axesOf(t), mk = MARKET[c.market];
  return h("li", { class: hidden ? "extra" : null },
    h("a", { href: "#/c/" + encodeURIComponent(t) },
      h("span", { class: "co-line" },
        h("b", { class: "co-name" }, koName(t)),
        codeTag(t),
        h("span", { class: "co-tags" },
          z.lead === t ? h("span", { class: "tag lead" }, "★ 대표") : null,
          uvCo(t) && uvCo(t).flag ? h("span", { class: "uv-tag sm" }, "저평가") : null,
          m.pos && z.lead !== t ? h("span", { class: "tag pos" }, m.pos) : null,
          mk && c.market !== "US" && c.market !== "private" ? h("span", { class: "tag mk" }, mk) : null,
          ax.has("openai") ? h("i", { class: "pip oa", title: "OpenAI와 계약·투자 관계" }) : null,
          ax.has("anthropic") ? h("i", { class: "pip an", title: "Anthropic과 계약·투자 관계" }) : null)),
      h("span", { class: "co-role" }, m.role)));
}

/* Stop 13's special zone: the two labs themselves. */
function axesPair() {
  return h("div", { class: "axes-pair" }, ["openai", "anthropic"].map((k) => {
    const a = D.axes[k], n = new Set(D.links.filter((l) => l.axis === k).map((l) => l.t)).size;
    return h("a", { class: `ax-card ${AX[k].cls}`, href: "#/axes" },
      h("b", null, a.name), h("span", null, a.products), h("span", { class: "ax-n" }, `계약·투자 상대 ${n}곳 →`));
  }));
}

/* ------------------------------------------------------------- diagrams (stops without a render) */
function dgText(x, y, str, cls, anchor = "middle") { return s("text", { x, y, class: cls, "text-anchor": anchor }, str); }
function dgBox(g, x, y, w, hgt, title, sub, opts = {}) {
  const grp = s("g", { class: "dg-box" + (opts.cls ? " " + opts.cls : "") });
  grp.append(s("rect", { x, y, width: w, height: hgt, rx: 10 }));
  grp.append(dgText(x + w / 2, y + (sub ? hgt / 2 - 3 : hgt / 2 + 5), title, "dg-t"));
  [].concat(sub || []).forEach((line, i) => grp.append(dgText(x + w / 2, y + hgt / 2 + 14 + i * 13, line, "dg-s")));
  if (opts.href) { const a = s("a", { href: opts.href }); a.append(grp); g.append(a); } else g.append(grp);
  return grp;
}
function dgArrow(g, x1, y1, x2, y2, label) {
  g.append(s("path", { d: `M${x1},${y1} L${x2},${y2}`, class: "dg-ln", "marker-end": "url(#dgArrow)" }));
  if (label) g.append(dgText(x1 + 8, (y1 + y2) / 2 + 4, label, "dg-l", "start"));
}
function dgRoot(w, hgt, label) {
  const root = s("svg", { viewBox: `0 0 ${w} ${hgt}`, class: "dg", role: "img", "aria-label": label });
  const mk = s("marker", { id: "dgArrow", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto" });
  mk.append(s("path", { d: "M0,0 L8,4 L0,8 Z", class: "dg-head" }));
  root.append(s("defs", null, mk));
  return root;
}
const DIAGRAMS = {
  cloud(st) {
    const g = dgRoot(360, 312, "클라우드가 데이터센터의 연산을 빌려주는 구조");
    dgBox(g, 20, 10, 320, 54, "AI 데이터센터 (11번)", "서버 수만 대 · 전기 · 냉각", { href: "#/s/datacenter" });
    dgArrow(g, 180, 64, 180, 96, "연산 자원");
    const zs = [["hyperscale", "하이퍼스케일러", ["AWS·애저", "구글·오라클"]], ["neocloud", "네오클라우드", ["코어위브", "네비우스"]], ["edge", "엣지", ["사용자 가까이", "클라우드플레어"]]];
    zs.forEach(([zid, t, sub], i) => dgBox(g, 20 + i * 110, 100, 100, 78, t, sub, { cls: "zone", href: `#/s/${st.id}/${zid}` }));
    dgArrow(g, 180, 178, 180, 212, "시간 단위로 빌려줌");
    dgBox(g, 20, 216, 320, 54, "고객", "AI 연구소 · 기업 · 개발자", { href: "#/s/labs" });
    g.append(dgText(180, 298, "큰 손님은 몇 년 치를 미리 계약 → 그 잔고가 미래 매출", "dg-note"));
    return g;
  },
  labs(st) {
    const g = dgRoot(360, 400, "AI 모델이 학습과 추론으로 만들어지고 쓰이는 과정");
    dgBox(g, 20, 10, 150, 52, "데이터", "글·코드·이미지");
    dgBox(g, 190, 10, 150, 52, "연산 (12번)", "칩 수십만 개", { href: "#/s/cloud" });
    g.append(s("path", { d: "M95,62 C95,84 180,76 180,98", class: "dg-ln" }));
    g.append(s("path", { d: "M265,62 C265,84 180,76 180,98", class: "dg-ln", "marker-end": "url(#dgArrow)" }));
    dgBox(g, 20, 100, 320, 66, "학습 = 공부", ["데이터를 보며 규칙을 익힙니다", "칩과 전기가 가장 많이 드는 단계"], { cls: "hot" });
    dgArrow(g, 180, 166, 180, 194);
    const cx = 180, cy = 224;
    g.append(s("circle", { cx, cy, r: 28, class: "dg-model" }));
    g.append(dgText(cx, cy + 5, "AI 모델", "dg-t inv"));
    const tag = (x, k, anchor) => {
      const a = s("a", { href: `#/s/${st.id}/axes` });
      a.append(s("circle", { cx: x, cy: cy - 6, r: 5, class: `dg-ax ${AX[k].cls}` }));
      a.append(dgText(anchor === "end" ? x - 10 : x + 10, cy - 2, D.axes[k].name, "dg-t sm", anchor));
      a.append(dgText(anchor === "end" ? x - 10 : x + 10, cy + 13, AX[k].prod, "dg-s", anchor));
      g.append(a);
    };
    tag(126, "openai", "end");
    tag(234, "anthropic", "start");
    dgArrow(g, 180, 252, 180, 280);
    dgBox(g, 20, 284, 320, 66, "추론 = 답하기", ["질문을 받아 답을 만듭니다", "매일 수억 명이 씁니다"], { cls: "cool" });
    dgArrow(g, 180, 350, 180, 374);
    g.append(dgText(180, 392, "앱·기기를 거쳐 우리에게 (14번)", "dg-note"));
    return g;
  },
  apps(st) {
    const g = dgRoot(360, 352, "AI 모델이 여러 갈래로 고객에게 닿는 길");
    const cx = 180, cy = 176, R = 122;
    st.zones.forEach((z, i) => {
      const ang = (-90 + i * (360 / st.zones.length)) * Math.PI / 180;
      const x = cx + R * Math.cos(ang), y = cy + R * Math.sin(ang);
      g.append(s("line", { x1: cx + 46 * Math.cos(ang), y1: cy + 46 * Math.sin(ang), x2: x - 24 * Math.cos(ang), y2: y - 24 * Math.sin(ang), class: "dg-ln" }));
      const a = s("a", { href: `#/s/${st.id}/${z.id}` });
      a.append(s("circle", { cx: x, cy: y, r: 22, class: "dg-node" }));
      const ic = icon(z.id, "dg-ic");
      ic.setAttribute("x", x - 11); ic.setAttribute("y", y - 11); ic.setAttribute("width", 22); ic.setAttribute("height", 22);
      a.append(ic);
      const ly = Math.sin(ang) < -0.3 ? y - 30 : y + 38;   // labels above the upper nodes, below the rest
      a.append(dgText(x, ly, z.name.replace(/\s*\(.*\)/, ""), "dg-s strong"));
      g.append(a);
    });
    g.append(s("circle", { cx, cy, r: 44, class: "dg-model" }));
    g.append(dgText(cx, cy - 2, "AI 모델", "dg-t inv"));
    g.append(dgText(cx, cy + 15, "13번에서", "dg-s inv"));
    return g;
  },
};

/* ------------------------------------------------------------- full cutaway viewer */
let viewerEl = null;
function openViewer(st, pid) {
  closeViewer();
  const p = D.panels[pid];
  let fit = store.get("fit", false);
  const canvas = h("div", { class: "vw-canvas", style: `--w:${p.w}px;aspect-ratio:${p.w}/${p.h}` },
    h("img", { src: p.img, alt: p.title, width: p.w, height: p.h, decoding: "async", draggable: "false" }),
    labels(st, p));
  const pan = h("div", { class: "vw-pan" }, canvas);
  const mode = h("button", { type: "button", class: "btn sm", onclick: () => { fit = !fit; store.set("fit", fit); apply(); } });
  const apply = () => {
    viewerEl.classList.toggle("fit", fit);
    mode.textContent = fit ? "원래 크기로" : "화면에 맞추기";
    if (!fit) centerOn();
  };
  const centerOn = () => {
    const first = p.spots.find((sp) => I.zoneById[sp.zone] && I.zoneById[sp.zone].st === st) || p.spots[0];
    if (!first) return;
    requestAnimationFrame(() => {
      pan.scrollLeft = first.anchor[0] - pan.clientWidth / 2;
      pan.scrollTop = first.anchor[1] - pan.clientHeight / 2;
    });
  };
  viewerEl = h("div", { class: "viewer", role: "dialog", "aria-modal": "true", "aria-label": `${p.title} 크게 보기` },
    h("header", { class: "vw-head" },
      h("div", null, h("b", null, p.title), h("span", null, p.scene)),
      mode,
      h("button", { type: "button", class: "vw-close", "aria-label": "닫기", onclick: () => back(`#/s/${st.id}`) }, "✕")),
    pan,
    h("footer", { class: "vw-foot" },
      p.legend.length ? h("span", { class: "legend" }, p.legend.map(([k, label]) => h("span", null, h("i", { style: `background:${INK[k]}` }), label))) : null,
      h("span", null, "금색 = 이 일터의 분야 · 이름표를 누르면 그 분야로 갑니다")));
  $("#layer").append(viewerEl);
  document.body.classList.add("locked");
  apply();
  viewerEl.querySelector(".vw-close").focus({ preventScroll: true });
}
function closeViewer() {
  if (!viewerEl) return;
  viewerEl.remove();
  viewerEl = null;
  document.body.classList.remove("locked");
}
function labels(st, p) {
  const ov = s("svg", { class: "vw-ov", viewBox: `0 0 ${p.w} ${p.h}` });
  const lines = s("g"), labs = s("g");
  ov.append(lines, labs);
  for (const sp of p.spots) {
    const hit = I.zoneById[sp.zone];
    if (!hit) continue;
    const here = hit.st === st ? " here" : "";
    const [ax, ay] = sp.anchor, [ex, ey] = sp.end, [x, y, w, hh] = sp.box;
    lines.append(s("line", { class: "ldr-u", x1: ax, y1: ay, x2: ex, y2: ey }), s("line", { class: "ldr" + here, x1: ax, y1: ay, x2: ex, y2: ey }),
      s("circle", { class: "pinpt" + here, cx: ax, cy: ay, r: 4.5 }));
    const a = s("a", { href: `#/s/${hit.st.id}/${sp.zone}`, class: "lab" + here, "data-left": String(ax > x + w / 2) });
    const g = s("g", { transform: `translate(${x},${y})` });
    g.append(s("rect", { width: w, height: hh, rx: 6 }));
    const t1 = s("text", { x: 7, y: 17 });
    t1.append(s("tspan", { class: "n" }, sp.text[0]), s("tspan", { class: "nm", dx: 5 }, sp.text[1]));
    g.append(t1);
    if (sp.text.length > 2 && sp.text[2]) g.append(s("text", { class: "sub", x: 7, y: 31 }, sp.text.slice(2).join(" · ")));
    a.append(g);
    labs.append(a);
    const big = s("text", { class: "fitnum" + here, x: ax, y: ay - 9, "text-anchor": "middle" }, sp.text[0]);
    lines.append(big);
  }
  requestAnimationFrame(() => fitLabels(ov));
  return ov;
}
/* The layout measured the label text with a width table; widen a box if this font runs longer. */
function fitLabels(ov) {
  if (!ov.isConnected) return;
  for (const a of ov.querySelectorAll(".lab")) {
    const g = a.firstChild, rect = g.querySelector("rect");
    let need = 0;
    for (const t of g.querySelectorAll("text")) need = Math.max(need, t.getComputedTextLength() + 14);
    const have = +rect.getAttribute("width");
    if (!need || need <= have) continue;
    rect.setAttribute("width", need.toFixed(1));
    if (a.dataset.left === "true") {
      const m = g.getAttribute("transform").match(/translate\(([-\d.]+),([-\d.]+)\)/);
      g.setAttribute("transform", `translate(${+m[1] - (need - have)},${m[2]})`);
    }
  }
}

/* ------------------------------------------------------------- glossary sheet */
let sheetEl = null;
function openTerm(g) {
  closeSheet();
  sheetEl = h("div", { class: "sheet", role: "dialog", "aria-label": g.term, onclick: (e) => { if (e.target === sheetEl) closeSheet(); } },
    h("div", { class: "sheet-card" },
      h("div", { class: "sheet-top" }, h("b", null, g.term), h("button", { type: "button", class: "vw-close", "aria-label": "닫기", onclick: closeSheet }, "✕")),
      h("p", null, g.short),
      g.stops && g.stops.length ? h("p", { class: "sheet-where" }, "나오는 곳 ",
        g.stops.map((id) => h("a", { class: "hop", href: "#/s/" + id }, `${I.stopById[id].no}. ${I.stopById[id].name}`))) : null,
      h("a", { class: "sheet-more", href: "#/words/" + encodeURIComponent(g.term) }, "용어 사전 전체 보기 →")));
  $("#layer").append(sheetEl);
  requestAnimationFrame(() => sheetEl && sheetEl.classList.add("open"));
}
function closeSheet() { if (sheetEl) { sheetEl.remove(); sheetEl = null; } }
addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (sheetEl) closeSheet();
  else if (viewerEl) viewerEl.querySelector(".vw-close").click();
});

/* ------------------------------------------------------------- company */
function viewCompany(t) {
  const c = co(t), here = I.spots[t] || [], links = I.links[t] || [];
  bar({ back: { href: "#/", label: "노선도" }, title: koName(t) });
  const stopsHere = new Set(here.map((x) => x.st.id));
  const mainZ = I.main[t];
  const peers = mainZ ? leadFirst(mainZ.z).filter((m) => m.t !== t) : [];
  mount($("#app"),
    h("article", { class: "company wrap" },
      h("p", { class: "kicker" }, c.market === "private" ? "비상장 회사" : `${MARKET[c.market] || c.market} 상장 · ${t}`),
      h("h1", { class: "co-title" }, koName(t)),
      c.name && c.name !== koName(t) ? h("p", { class: "co-en" }, c.name) : null,
      h("div", { class: "strip", role: "img", "aria-label": `노선도에서 이 회사가 있는 일터: ${[...stopsHere].map((id) => I.stopById[id].no).join(", ") || "없음"}` },
        D.stops.map((st) => h("a", { href: "#/s/" + st.id, class: "strip-dot" + (stopsHere.has(st.id) ? " on" : "") + (mainZ && mainZ.st === st ? " main" : ""), title: `${st.no}. ${st.name}` }, st.no))),
      here.length ? h("section", { class: "co-sec" },
        h("h2", { class: "sec" }, "이 회사가 하는 일"),
        h("ul", { class: "roles" }, here.map((x) => h("li", null,
          h("a", { class: "role-where", href: `#/s/${x.st.id}/${x.z.id}` }, `${x.st.no}. ${x.st.name}`, h("span", null, " › "), x.z.name),
          h("div", { class: "role-tags" },
            x.z.lead === t ? h("span", { class: "tag lead" }, "★ 대표") : null,
            x.m.pos ? h("span", { class: "tag pos" }, x.m.pos) : null,
            mainZ && mainZ.z === x.z && here.length > 1 ? h("span", { class: "tag" }, "대표 분야") : null),
          h("p", null, x.m.role, x.m.src ? src(x.m.src) : null),
          x.z.lead === t && x.z.lead_why !== x.m.role ? h("p", { class: "muted" }, `대표로 고른 이유: ${x.z.lead_why}`) : null)))) : null,
      uvCompany(t),
      links.length ? h("section", { class: "co-sec" },
        h("h2", { class: "sec" }, "OpenAI·Anthropic과의 관계", h("small", null, `${links.length}건`)),
        h("ul", { class: "deals" }, links.map(dealLi))) : null,
      !here.length && !links.length ? h("p", null, "지도에 이 회사의 기록이 없습니다.") : null,
      peers.length ? h("section", { class: "co-sec" },
        h("h2", { class: "sec" }, "같은 분야의 회사", h("small", null, mainZ.z.name)),
        h("div", { class: "chips" }, peers.map((m) => h("a", { class: "chip", href: "#/c/" + encodeURIComponent(m.t) },
          m.t === mainZ.z.lead ? "★ " : "", koName(m.t))))) : null),
    foot());
}
function dealLi(ln, withCompany = false) {
  const a = AX[ln.axis], size = [ln.gw ? `${ln.gw}GW` : null, ln.usd_bn ? usd(ln.usd_bn) : null].filter(Boolean).join(" · ");
  return h("li", { class: `deal st-${ln.status}` },
    h("span", { class: `pip ${a.cls}`, "aria-hidden": "true" }),
    h("div", null,
      h("p", { class: "deal-top" },
        withCompany ? [h("a", { class: "deal-co", href: "#/c/" + encodeURIComponent(ln.t) }, koName(ln.t)), h("span", { class: "deal-ax" }, `↔ ${a.ko}`)]
          : h("b", null, a.ko),
        h("span", { class: "tag kind" }, ln.kind),
        h("span", { class: `tag status s-${ln.status}`, title: D.statuses[ln.status] || "" }, ln.status)),
      h("p", { class: "deal-what" }, ln.what),
      h("p", { class: "deal-meta" }, [ln.date.slice(0, 7), size].filter(Boolean).join(" · "),
        ln.update ? h("span", { class: "upd" }, ` → ${ln.update}`) : null, " ", src(ln.src))));
}

/* ------------------------------------------------------------- find */
const norm = (x) => (x || "").toLowerCase().replace(/[\s·()\-.,]/g, "");
const MARKET_FILTERS = [["all", "전체"], ["US", "미국"], ["KR", "한국"], ["JP", "일본"], ["TW", "대만"], ["CHN", "중국·홍콩"], ["EUIN", "유럽·인도"], ["private", "비상장"]];
function viewFind() {
  bar({ back: { href: "#/", label: "노선도" }, title: "찾기" });
  let market = "all";
  const all = Object.keys(D.companies).filter((t) => I.spots[t] || I.links[t])
    .sort((a, b) => koName(a).localeCompare(koName(b), "ko"));
  const input = h("input", { type: "search", class: "find-input", placeholder: "회사·분야·용어 (예: 엔비디아, HBM, 005930)", "aria-label": "검색어", autocomplete: "off", enterkeyhint: "search" });
  const chips = h("div", { class: "chips filters", role: "radiogroup", "aria-label": "시장" });
  const out = h("div", { class: "find-out" });
  const drawChips = () => mount(chips, MARKET_FILTERS.map(([k, label]) => h("button", { type: "button", role: "radio", "aria-checked": String(market === k), class: "chip", onclick: () => { market = k; drawChips(); draw(); } }, label)));
  const inMarket = (t) => {
    const m = co(t).market;
    return market === "all" || m === market || (market === "CHN" && (m === "CN" || m === "HK")) || (market === "EUIN" && (m === "EU" || m === "IN"));
  };
  const draw = () => {
    const q = norm(input.value);
    if (!q) {
      const list = all.filter(inMarket);
      mount(out, h("p", { class: "muted" }, `${list.length}개 회사 · 가나다순`), h("ul", { class: "co-list" }, list.map(coItem)));
      return;
    }
    const named = (t) => [t, co(t).ko, co(t).short, co(t).name, ...(co(t).aliases || [])].some((x) => norm(x).includes(q));
    const cos = all.filter((t) => inMarket(t) && named(t));
    const roles = q.length < 2 ? [] : all.filter((t) => inMarket(t) && !named(t)).flatMap((t) => {
      const x = (I.spots[t] || []).find((y) => norm(y.m.role).includes(q));
      return x ? [x] : [];
    });
    const zones = Object.values(I.zoneById).filter(({ z }) => norm(z.name).includes(q));
    const words = D.glossary.filter((g) => [g.term, ...g.match].some((x) => norm(x).includes(q)));
    mount(out,
      zones.length ? [h("h2", { class: "sec" }, "분야"), h("ul", { class: "co-list" }, zones.map(({ st, z }) => h("li", null, h("a", { href: `#/s/${st.id}/${z.id}` }, h("b", null, z.name), h("span", { class: "muted" }, ` ${st.no}. ${st.name} · 회사 ${z.members.length}곳`)))))] : null,
      words.length ? [h("h2", { class: "sec" }, "용어"), h("ul", { class: "co-list" }, words.map((g) => h("li", null, h("a", { href: "#/words/" + encodeURIComponent(g.term) }, h("b", null, g.term), h("span", { class: "muted" }, " " + g.short)))))] : null,
      cos.length || !roles.length ? [h("h2", { class: "sec" }, "회사", h("small", null, `${cos.length}곳`)),
        cos.length ? h("ul", { class: "co-list" }, cos.map(coItem)) : h("p", { class: "muted" }, "이름이 맞는 회사가 없습니다. 영어 이름이나 티커로도 찾아 보세요.")] : null,
      roles.length ? [h("h2", { class: "sec" }, "하는 일에 이 말이 나오는 회사", h("small", null, `${roles.length}곳`)),
        h("ul", { class: "co-list" }, roles.map((x) => h("li", null, h("a", { href: "#/c/" + encodeURIComponent(x.m.t) },
          h("b", null, koName(x.m.t)), codeTag(x.m.t), h("span", { class: "muted" }, `${x.st.no}. ${x.st.name} · ${x.z.name} — ${x.m.role}`)))))] : null);
  };
  let timer = 0;
  input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(draw, 80); });
  mount($("#app"), h("section", { class: "find wrap" }, h("h1", { class: "page-title" }, "회사·분야·용어 찾기"), input, chips, out), foot());
  drawChips();
  draw();
  setTimeout(() => input.focus({ preventScroll: true }), 50);
}
function coItem(t) {
  const x = I.main[t];
  return h("li", null, h("a", { href: "#/c/" + encodeURIComponent(t) },
    h("b", null, koName(t)), codeTag(t),
    h("span", { class: "muted" }, x ? ` ${x.st.no}. ${x.st.name} · ${x.z.name}` : " 두 축 계약 상대")));
}

/* ------------------------------------------------------------- words */
function viewWords() {
  bar({ back: { href: "#/", label: "노선도" }, title: "용어 사전" });
  const order = (g) => { const st = g.stops && g.stops.length ? I.stopById[g.stops[0]] : null; return st ? st.no : 99; };
  const list = [...D.glossary].sort((a, b) => order(a) - order(b) || a.term.localeCompare(b.term, "ko"));
  let group = null;
  const items = [];
  for (const g of list) {
    const st = g.stops && g.stops.length ? I.stopById[g.stops[0]] : null, label = st ? `${st.no}. ${st.name}` : "두루 쓰는 말";
    if (label !== group) { group = label; items.push(h("li", { class: "w-group" }, label)); }
    items.push(h("li", { class: "word", id: "w-" + slug(g.term) },
      h("b", null, g.term),
      h("p", null, g.short),
      g.stops && g.stops.length ? h("p", { class: "sheet-where" }, g.stops.map((id) => h("a", { class: "hop", href: "#/s/" + id }, `${I.stopById[id].no}. ${I.stopById[id].name}`))) : null));
  }
  mount($("#app"), h("section", { class: "words wrap" }, h("h1", { class: "page-title" }, "용어 사전"),
    h("p", { class: "lede" }, `본문에서 점선 밑줄이 있는 말을 누르면 같은 설명이 뜹니다. 처음 나오는 일터 순서로 모았습니다 · ${D.glossary.length}개`),
    h("ul", { class: "word-list" }, items)), foot());
}

/* ------------------------------------------------------------- axes */
function viewAxes() {
  bar({ back: { href: "#/", label: "노선도" }, title: "OpenAI·Anthropic" });
  let pick = "all";
  const body = h("div");
  const kinds = Object.keys(D.kinds);
  const seg = h("div", { class: "chips filters", role: "radiogroup", "aria-label": "회사 고르기" });
  const draw = () => {
    mount(seg, [["all", "둘 다"], ["openai", "OpenAI"], ["anthropic", "Anthropic"]].map(([k, label]) =>
      h("button", { type: "button", class: "chip", role: "radio", "aria-checked": String(pick === k), onclick: () => { pick = k; draw(); } },
        k !== "all" ? h("i", { class: `pip ${AX[k].cls}` }) : null, label)));
    const rows = D.links.filter((l) => pick === "all" || l.axis === pick);
    mount(body, kinds.map((k) => {
      const list = rows.filter((l) => l.kind === k).sort((a, b) => b.date.localeCompare(a.date));
      if (!list.length) return null;
      return h("section", { class: "co-sec" },
        h("h2", { class: "sec" }, k, h("small", null, `${D.kinds[k]} · ${list.length}건`)),
        h("ul", { class: "deals" }, list.map((ln) => dealLi(ln, true))));
    }));
  };
  mount($("#app"),
    h("section", { class: "axes wrap" },
      h("p", { class: "kicker" }, "13번 일터 · AI 모델"),
      h("h1", { class: "page-title" }, "OpenAI·Anthropic의 돈 흐름"),
      h("p", { class: "lede" }, rich(I.zoneById.axes ? I.zoneById.axes.z.need : "")),
      h("div", { class: "ax-cards" }, ["openai", "anthropic"].map((k) => {
        const a = D.axes[k];
        return h("article", { class: `ax-full ${AX[k].cls}` },
          h("header", null, h("b", null, a.name), h("span", null, a.products), h("span", { class: "tag" }, "비상장")),
          h("p", null, a.summary),
          h("dl", null, a.facts.map((f) => [h("dt", null, f.label), h("dd", null, f.value, h("span", { class: "muted" }, ` ${f.date.slice(0, 7)} `), src(f.src))])));
      })),
      h("p", { class: "note" }, "상태: ", Object.entries(D.statuses).map(([k, v]) => h("span", { class: `tag status s-${k}`, title: v }, k)), " — 금액과 GW는 서로 겹칠 수 있어(예: 엔비디아 칩이 오라클 데이터센터에 들어감) 합계를 내지 않습니다."),
      seg, body),
    foot());
  draw();
}

/* ------------------------------------------------------------- 저평가 기준 */
function viewValue() {
  bar({ back: { href: "#/", label: "노선도" }, title: "저평가 기준" });
  const V = D.value;
  if (!V) {
    mount($("#app"), h("section", { class: "wrap" }, h("h1", { class: "page-title" }, "저평가 표시"),
      h("p", { class: "lede" }, "이번 빌드에는 저평가 자료가 없습니다.")), foot());
    return;
  }
  const c = V.check, all = uvZonesAll(), flagged = all.filter((x) => x.v.flag);
  const cos = Object.entries(V.companies).filter(([, x]) => x.flag).sort((a, b) => a[1].V - b[1].V);
  mount($("#app"),
    h("article", { class: "wrap value-page" },
      h("p", { class: "kicker" }, `저평가 표시 기준 · ${V.price_date} 종가`),
      h("h1", { class: "page-title" }, "싸 보이는 것과 싸진 것은 다릅니다"),
      h("p", { class: "lede" }, "PER이 낮다고 싼 것은 아닙니다. 메모리처럼 이익이 정점인 업종은 PER이 늘 낮아 보이고, 미래 이익만 보면 언제나 메모리를 사야 할 것처럼 보입니다. 그래서 이 사이트는 회사마다 '자기 이익 전망에 비해 주가가 얼마나 덜 올랐나'를 재고, 그 변화를 분야끼리 비교합니다. 돈이 분야 사이를 옮겨 다닌다는(순환매) 전제입니다."),
      h("ol", { class: "uv-steps" },
        h("li", null, h("b", null, "이익 전망 변화"), ` 최근 ${V.window.days}일 동안 다음 회계연도 EPS 컨센서스가 몇 % 바뀌었나`),
        h("li", null, h("b", null, "주가 변화"), " 같은 기간 주가가 몇 % 바뀌었나"),
        h("li", null, h("b", null, "선행 PER 변화"), " = (1 + 주가 변화) ÷ (1 + 전망 변화) − 1. 마이너스면 전망만큼 주가가 따라가지 못해 이익 대비 싸진 것입니다"),
        h("li", null, h("b", null, "저평가 분야"), " 분야 중앙값을 AI 전체와 비교해 가장 많이 싸진 1/3 가운데, 전망은 오르고 PER은 실제로 내린 분야. 흑자이고 애널리스트가 3명 이상인 회사가 3곳 이상인 분야만 잽니다"),
        h("li", null, h("b", null, "저평가 회사"), " 저평가 분야가 대표 분야이고, 자기 전망은 오르고 PER은 내린 회사")),
      h("h2", { class: "sec" }, "지금 저평가 분야", h("small", null, `${flagged.length}곳 · AI 전체 PER ${pct(V.universe.V)}`)),
      flagged.length ? uvRows(flagged) : h("p", { class: "muted" }, "지금은 조건을 채운 분야가 없습니다."),
      h("h2", { class: "sec" }, "지금 저평가 회사", h("small", null, `${cos.length}곳`)),
      cos.length ? h("ul", { class: "co-list" }, cos.map(([t, x]) => {
        const hit = I.zoneById[x.zone];
        return h("li", null, h("a", { href: "#/c/" + encodeURIComponent(t) }, h("b", null, koName(t)), codeTag(t),
          h("span", { class: "muted" }, `${hit ? hit.z.name : ""} — 전망 ${pct(x.R)} · 주가 ${pct(x.P)} · PER ${pct(x.V)}`)));
      })) : h("p", { class: "muted" }, "없습니다."),
      h("h2", { class: "sec" }, "2026년에 맞았나"),
      h("p", null, `그때 미리 알 수 있던 전망만으로 ${c.period}의 두 달을 다시 재 봤습니다. 저평가 분야는 ${c.horizon} 동안 AI 전체(중앙값)보다 ${c.zone_flag.map(pp).join(" · ")}%p 더 올랐고 분야의 ${c.zone_hit.join(" · ")}%가 앞섰습니다. 가장 비싸진 1/3은 ${c.zone_dear.map(pp).join(" · ")}%p였습니다.`),
      h("p", null, `저평가 회사는 평균 ${c.company_mean.map(pp).join(" · ")}%p, 중앙값 ${c.company_median.map(pp).join(" · ")}%p였고 ${c.company_hit.join(" · ")}%가 앞섰습니다. ${c.fades}. 지금 PER 수준, 성장 대비 PER, 전망 상향만으로 고른 방식은 같은 시험에서 앞서지 못해 쓰지 않았습니다.`),
      h("h2", { class: "sec" }, "한계"),
      h("ul", { class: "how" },
        h("li", null, "검증은 두 달뿐이고 여러 방식 중에서 골랐습니다. 효과가 작아서 회사 하나하나로는 동전 던지기에 가깝습니다. 분야 단위로 보세요."),
        h("li", null, CYCLE_NOTE),
        h("li", null, "적자 회사, 애널리스트 3명 미만, 전망 자료가 없는 회사는 판단하지 않습니다. 전망은 Yahoo 집계라 국내 증권사 컨센서스와 다를 수 있습니다."),
        h("li", null, `${V.price_date} 종가로 계산한 한 달 단위 관찰입니다. 매수 추천이 아닙니다.`)),
      h("h2", { class: "sec" }, "모든 분야", h("small", null, "가장 많이 싸진 순서")),
      uvRows(all)),
    foot());
}

/* ------------------------------------------------------------- quiz */
function viewQuiz() {
  bar({ back: { href: "#/", label: "노선도" }, title: "퀴즈" });
  const score = { right: 0, total: 0 };
  const host = h("div", { class: "quiz-host" });
  const board = h("p", { class: "quiz-score" });
  const pool = Object.keys(I.main);
  const pickN = (arr, n, not) => {
    const out = [], copy = arr.filter((x) => !not.includes(x));
    while (out.length < n && copy.length) out.push(copy.splice(Math.floor(Math.random() * copy.length), 1)[0]);
    return out;
  };
  const shuffle = (arr) => { for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; };
  const ask = () => {
    board.textContent = score.total ? `${score.total}문제 중 ${score.right}개 맞힘` : "한 문제씩 풀어 보세요";
    const leadQ = Math.random() < 0.4;
    let q, options, answer, explain;
    if (leadQ) {
      const zs = Object.values(I.zoneById).filter(({ z }) => z.lead && z.members.length >= 2);
      const { st, z } = zs[Math.floor(Math.random() * zs.length)];
      answer = z.lead;
      const same = z.members.map((m) => m.t).filter((t) => t !== z.lead);
      const wrong = pickN(same, 3, []);
      if (wrong.length < 3) wrong.push(...pickN(pool, 3 - wrong.length, [answer, ...wrong]));
      options = shuffle([answer, ...wrong]).map((t) => ({ key: t, label: koName(t), sub: code(t) }));
      q = [h("span", { class: "q-kind" }, "대표 기업 맞히기"), h("b", null, `「${z.name}」 분야의 ★ 대표 기업은?`), h("span", { class: "q-hint" }, `${st.no}. ${st.name} · ${z.easy}`)];
      explain = () => [h("b", null, `${koName(z.lead)} (${code(z.lead)})`), h("p", null, z.lead_why), h("a", { class: "hop", href: `#/s/${st.id}/${z.id}` }, `${st.no}. ${st.name} › ${z.name} 읽기`)];
    } else {
      const t = pool[Math.floor(Math.random() * pool.length)], x = I.main[t];
      answer = x.st.id;
      const its = I.spots[t].map((y) => y.st.id);   // a company met at several stops: none of them is a wrong answer
      const wrong = pickN(D.stops.map((st) => st.id), 3, its);
      options = shuffle([answer, ...wrong]).map((id) => ({ key: id, label: `${I.stopById[id].no}. ${I.stopById[id].name}`, sub: I.stopById[id].makes }));
      q = [h("span", { class: "q-kind" }, "일터 맞히기"), h("b", null, `${koName(t)}(${code(t)})는 어느 일터에서 일할까?`)];
      explain = () => [h("b", null, `${x.st.no}. ${x.st.name} › ${x.z.name}`), h("p", null, x.m.role), h("a", { class: "hop", href: "#/c/" + encodeURIComponent(t) }, `${koName(t)} 자세히`)];
    }
    const res = h("div", { class: "q-res", "aria-live": "polite" });
    const opts = h("div", { class: "q-opts" }, options.map((o) => h("button", { type: "button", class: "q-opt", onclick: (e) => {
      if (host.dataset.done) return;
      host.dataset.done = "1";
      score.total += 1;
      const ok = o.key === answer;
      if (ok) score.right += 1;
      for (const b of opts.children) b.classList.add(b.dataset.key === answer ? "right" : b === e.currentTarget ? "wrong" : "off");
      mount(res, h("p", { class: ok ? "ok" : "no" }, ok ? "맞았습니다!" : "아쉬워요. 정답은"), explain(),
        h("button", { type: "button", class: "btn primary", onclick: () => { delete host.dataset.done; ask(); } }, "다음 문제 →"));
      board.textContent = `${score.total}문제 중 ${score.right}개 맞힘`;
    }, "data-key": o.key }, h("b", null, o.label), h("span", null, o.sub))));
    mount(host, h("div", { class: "q-card" }, h("p", { class: "q" }, q), opts, res));
  };
  mount($("#app"), h("section", { class: "quiz wrap" }, h("h1", { class: "page-title" }, "퀴즈로 익히기"), board, host), foot());
  ask();
}

/* ------------------------------------------------------------- boot */
fetch(`data.json?v=${V}`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }).then((d) => {
  D = d;
  I = index(d);
  route();
}).catch((e) => {
  mount($("#app"), h("p", { class: "boot" }, "지도를 불러오지 못했습니다. 잠시 뒤 새로고침해 주세요. (" + e.message + ")"));
});
if ("serviceWorker" in navigator && location.protocol === "https:") navigator.serviceWorker.register("sw.js").catch(() => {});
