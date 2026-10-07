"use strict";
/* AI 탐방 — the research desk visits the AI supply chain in product order
   (/api/tour = desk/tour_map.json + latest fund PDFs). A process-flow chart shows how power,
   materials and tools become chips, servers, a data center and finally the finished model and
   the apps that sell it; the numbered stops are the visit order. Each visit report says what
   the stop receives, makes and passes on, opens an illustrated site whose hotspots pick a
   department, and lists the companies met there. "보유 종목의 자리" places every fund or
   account holding on that flow. Shares app.js's helpers (h, svg, mount, api, S) and its tab
   router ("desk:tab"); nothing is drawn until the tab is opened. Rules: docs/desk_system.md §8. */

const TR = { data: null, idx: null, loading: null, stop: null, zone: null, focus: null,
             axis: "all", lens: "time", ledgerAxis: "all", ledgerKind: "all", visited: new Set() };
const TR_AXES = [["all", "전체"], ["openai", "OpenAI"], ["anthropic", "Anthropic"], ["both", "양쪽 모두"]];
const TR_LENSES = [["time", "TIME"], ["koact", "KoAct"], ["seven", "7종목"], ["mine", "내 보유"],
                   ["momentum", "수익률 상위 15"], ["none", "표시 끄기"]];
const TR_LENS_KO = { time: "TIME 비중", koact: "KoAct 비중", seven: "7종목 목표 비중", mine: "내 계좌 비중",
                     momentum: "120일 수익률 상위 15종목", none: "" };
const TR_AX = { openai: { ko: "OpenAI", cls: "oa" }, anthropic: { ko: "Anthropic", cls: "an" } };
const TR_STATUS = { 확정: "확정", 의향: "의향서(LOI)", 보도: "언론 보도", 확대: "▲ 확대", 축소: "▼ 축소", 종료: "종료" };
const TR_KINDS = ["투자", "클라우드", "칩", "메모리", "서버", "네트워크", "데이터센터", "전력", "유통"];

const trStore = {
  get(k, d) { try { return localStorage.getItem("tour-" + k) ?? d; } catch (_) { return d; } },
  set(k, v) { try { localStorage.setItem("tour-" + k, v); } catch (_) { /* storage unavailable */ } },
};
if (TR_AXES.some(([k]) => k === trStore.get("axis"))) TR.axis = trStore.get("axis");
if (TR_LENSES.some(([k]) => k === trStore.get("lens"))) TR.lens = trStore.get("lens");
TR.stop = trStore.get("stop", null);
try { TR.visited = new Set(JSON.parse(trStore.get("visited", "[]"))); } catch (_) { TR.visited = new Set(); }

/* ------------------------------------------------------------------ data */
function trIndex(d) {
  const stopById = {}, zoneById = {}, spots = {}, main = {}, links = {}, axes = {}, cells = {};
  for (const s of d.map.stops) {
    stopById[s.id] = s;
    cells[s.at.join(",")] = s;
    for (const z of s.zones) {
      zoneById[z.id] = { s, z };
      for (const mem of z.members || []) (spots[mem.t] ||= []).push({ s, z, mem });
    }
  }
  for (const [t, list] of Object.entries(spots)) main[t] = list.find((x) => x.mem.main) || list[0];
  for (const ln of d.map.links) (links[ln.t] ||= []).push(ln);
  for (const [t, list] of Object.entries(links)) {
    list.sort((a, b) => b.date.localeCompare(a.date));
    axes[t] = new Set(list.map((l) => l.axis));
  }
  return { stopById, zoneById, spots, main, links, axes, cells };
}
const trCo = (t) => TR.data.map.companies[t] || { name: t, ko: t, market: "US" };
const trShort = (t) => trCo(t).short || t;
const trSub = (t) => (/^[ -~]+$/.test(trShort(t)) && trCo(t).ko !== trShort(t) ? trCo(t).ko : "");  // Korean name beside a ticker
const trAxes = (t) => TR.idx.axes[t] || new Set();
const trNorm = (tk) => (TR.data.alias || {})[tk] || tk;
const trUsd = (bn) => (bn >= 1000 ? `${+(bn / 1000).toFixed(bn % 1000 ? 1 : 0)}조 달러`
  : `${Math.round(bn * 10).toLocaleString("ko-KR")}억 달러`);

function trLensMap(lens = TR.lens) {
  const f = TR.data.funds;
  if (lens === "time") return (f.TIME || {}).weights || {};
  if (lens === "koact") return (f.KoAct || {}).weights || {};
  if (typeof S === "undefined" || !S) return {};
  if (lens === "seven") return Object.fromEntries(S.targets.map((r) => [trNorm(r.ticker), r.weight]));
  if (lens === "mine") return Object.fromEntries(S.account.positions.map((p) => [trNorm(p.ticker), p.weight || 0]));
  if (lens === "momentum") return Object.fromEntries(S.rankings.map((r) => [trNorm(r.ticker), (16 - r.rank) / 15]));
  return {};
}
const trRank = (t) => (typeof S !== "undefined" && S ? S.rankings.find((r) => trNorm(r.ticker) === t) : null);

function trAxisOn(t) {  // null = no axis lens; true/false = matches the chosen axis
  if (TR.axis === "all") return null;
  const ax = trAxes(t);
  return TR.axis === "both" ? ax.has("openai") && ax.has("anthropic") : ax.has(TR.axis);
}
function trZoneAxes(z) {
  const out = new Set();
  for (const mem of z.members || []) for (const a of trAxes(mem.t)) out.add(a);
  return out;
}
const trStopCos = (s) => s.zones.flatMap((z) => (z.members || []).filter((m) => TR.idx.main[m.t].z === z).map((m) => m.t));
function trSum(ts, lensMap) {
  return TR.lens === "momentum" ? ts.filter((t) => t in lensMap).length : ts.reduce((s, t) => s + (lensMap[t] || 0), 0);
}
const trSumText = (v) => (TR.lens === "momentum" ? `${v}개` : pctAbs(v, 1));

async function trLoad(force) {
  if (TR.data && !force) { trRenderAll(); return; }
  if (TR.loading) return TR.loading;
  if (!TR.data) mount($("#tr-visit"), h("p", { class: "muted" }, "탐방 지도를 불러오는 중…"));
  TR.loading = api("/api/tour").then((d) => {
    TR.data = d;
    TR.idx = trIndex(d);
    if (!TR.idx.stopById[TR.stop]) TR.stop = d.map.stops[0].id;
    trVisited(TR.stop);
    trRenderAll();
  }).catch((e) => {
    // A desk started before this tab existed serves the new page but not /api/tour.
    const old = /not found/i.test(e.message);
    mount($("#tr-visit"), h("p", { class: "err" }, old
      ? "AI 산업 지도를 열려면 데스크를 재시작하세요. 실행 창을 닫고 ‘야간데스크.cmd’를 다시 실행하면 됩니다."
      : "탐방 지도를 불러오지 못했습니다. " + e.message));
  }).finally(() => { TR.loading = null; });
  return TR.loading;
}

function trRenderAll() {
  const d = TR.data;
  $("#tr-asof").textContent = d.map.as_of;
  $("#tr-pdf").textContent = ["TIME", "KoAct"].filter((k) => d.funds[k]).map((k) => `${k} ${md(d.funds[k].date)}`).join(" · ") || "—";
  trControls(); trRenderAlerts(); trRenderFlow(); trRenderLeads(); trRenderVisit(); trRenderMine(); trRenderAxes(); trRenderNotes();
}

/* --------------------------------------------------------------- controls */
function trControls() {
  const dot = (k) => (k === "openai" ? h("i", { class: "dot", style: "background:var(--ax-oa)" })
    : k === "anthropic" ? h("i", { class: "dot", style: "background:var(--ax-an)" })
      : k === "both" ? h("i", { class: "dot split" }) : null);
  mount($("#tr-axis"), TR_AXES.map(([k, label]) => h("button", {
    role: "radio", "aria-checked": String(TR.axis === k), onclick: () => trSetAxis(k),
    title: k === "all" ? "모든 기업을 표시합니다" : k === "both" ? "두 회사 모두와 계약·투자 관계가 있는 기업을 강조합니다" : `${label}와 계약·투자 관계가 있는 기업을 강조합니다`,
  }, dot(k), label)));
  const noDesk = typeof S === "undefined" || !S;
  mount($("#tr-lens"), TR_LENSES.map(([k, label]) => h("button", {
    role: "radio", "aria-checked": String(TR.lens === k), onclick: () => trSetLens(k),
    disabled: noDesk && ["seven", "mine", "momentum"].includes(k) ? true : null,
    title: k === "none" ? "보유 종목 강조 표시를 끕니다" : `표시 기준: ${TR_LENS_KO[k]}`,
  }, k !== "none" ? h("i", { class: "lampdot" }) : null, label)));
  const m = TR.data.map;
  mount($("#tr-names"), Object.entries(m.companies).map(([t, c]) => h("option", { value: `${t} ${c.ko}` })));
}
function trSetAxis(k) { TR.axis = k; trStore.set("axis", k); trControls(); trRenderFlow(); trRenderLeads(); trRenderVisit(); trRenderMine(); trRenderAxes(); }
function trSetLens(k) { TR.lens = k; trStore.set("lens", k); trControls(); trRenderFlow(); trRenderLeads(); trRenderVisit(); trRenderMine(); trRenderAxes(); }

function trFind(q) {
  q = (q || "").trim();
  if (!q) return;
  const m = TR.data.map, first = q.split(/\s+/)[0].toUpperCase(), low = q.toLowerCase();
  let t = m.companies[first] ? first : trNorm(first) in m.companies ? trNorm(first) : null;
  if (!t) {
    const hit = Object.entries(m.companies).find(([k, c]) =>
      [k, c.ko, c.name, c.short].some((s) => s && s.toLowerCase().includes(low)));
    t = hit && hit[0];
  }
  if (!t) { toast(`‘${q}’를 지도에서 찾지 못했습니다`); return; }
  trFocus(t);
}

/* ---------------------------------------------------------------- alerts */
function trRenderAlerts() {
  const d = TR.data, out = [];
  const banner = (text) => h("div", { class: "banner" }, h("span", { class: "ico", "aria-hidden": "true" }, "!"), h("span", null, text));
  if (d.problems.length)
    out.push(banner(`탐방 지도에서 확인이 필요한 항목 ${d.problems.length}건을 발견했습니다: ${d.problems.slice(0, 3).join(" · ")}${d.problems.length > 3 ? " …" : ""}`));
  if (d.stale)
    out.push(banner(`지도 기준일(${d.map.as_of})로부터 ${d.age_days}일이 지났습니다. 계약 관계를 다시 확인하도록 ‘AI 산업 지도 갱신’을 요청하세요. 점검 주기는 ${d.map.review_days}일입니다.`));
  if ((d.universe_missing || []).length)
    out.push(banner(`AI 분석 대상 중 ${d.universe_missing.length}종목이 탐방 지도에 없습니다: ${d.universe_missing.slice(0, 8).join(", ")}. ‘AI 산업 지도 갱신’을 요청하면 방문지를 배정합니다.`));
  if (d.unplaced.length)
    out.push(banner(`두 펀드의 새 보유 종목 중 ${d.unplaced.length}종목이 아직 탐방 지도에 없습니다: ${d.unplaced.slice(0, 6).map((r) => `${r.t}(${r.name})`).join(", ")}. ‘AI 산업 지도 갱신’을 요청하면 방문지를 배정합니다.`));
  mount($("#tr-alerts"), out);
}

/* ------------------------------------------------------------- companies */
function trWin(t, lensMap, maxW) {
  const c = trCo(t), ax = trAxes(t), w = TR.lens === "none" ? undefined : lensMap[t], on = trAxisOn(t);
  const cls = ["win"];
  if (c.market === "private") cls.push("private");
  if (/[가-힣]/.test(trShort(t))) cls.push("kr");
  if (w != null) cls.push("lit");
  if (on === false) cls.push("dim");
  if (on === true) cls.push(TR.axis === "both" ? "ring-both" : TR.axis === "openai" ? "ring-oa" : "ring-an");
  if (TR.focus === t) cls.push("found");
  const l = w != null ? Math.max(0.3, Math.min(1, maxW > 0 ? w / maxW : 1)) : null;
  return h("span", { class: cls.join(" "), "data-t": t, style: l != null ? `--l:${l.toFixed(2)}` : null },
    h("span", null, trShort(t)),
    ax.has("openai") ? h("i", { class: "pip oa" }) : null,
    ax.has("anthropic") ? h("i", { class: "pip an" }) : null);
}

function trSpires() {
  const W = 360, H = 150;
  const root = svg("svg", { class: "tr-spires", viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "AI 모델 개발사: OpenAI와 Anthropic" });
  const spire = (axis, cx, name, prod, side) => {
    const a = TR_AX[axis], active = TR.axis === axis;
    const g = svg("g", { class: `tr-spire ${a.cls}${active ? " active" : ""}`, tabindex: "0", role: "button",
                         "aria-pressed": String(active), "aria-label": `${name}와 연결된 기업 강조` }, root);
    const base = H - 4, top = 56, wB = 66, wT = 34;
    svg("path", { class: "body", d: `M${cx - wB / 2},${base} L${cx - wT / 2},${top} L${cx + wT / 2},${top} L${cx + wB / 2},${base} Z` }, g);
    svg("path", { class: "crown", d: `M${cx - wT / 2 - 3},${top} L${cx},${top - 18} L${cx + wT / 2 + 3},${top} Z` }, g);
    svg("line", { class: "mast", x1: cx, y1: top - 18, x2: cx, y2: 12 }, g);
    svg("circle", { class: "halo", cx, cy: 12, r: 7 }, g);
    svg("circle", { class: "beacon", cx, cy: 12, r: 3 }, g);
    const lit = active || TR.axis === "both";
    for (let y = top + 8, row = 0; y < base - 6; y += 9, row++) {
      const half = (wT + (wB - wT) * ((y - top) / (base - top))) / 2 - 5;
      const cols = Math.max(2, Math.floor((half * 2) / 8)), cw = (half * 2) / cols;
      for (let c = 0; c < cols; c++)
        svg("rect", { class: "sw" + (lit && (row * 7 + c * 3) % 5 ? " on" : ""), x: cx - half + c * cw + 1, y, width: cw - 2.5, height: 5 }, g);
    }
    const lx = side === "left" ? cx - wB / 2 - 8 : cx + wB / 2 + 8, anchor = side === "left" ? "end" : "start";
    const t1 = svg("text", { class: "nm", x: lx, y: 96, "text-anchor": anchor }, g); t1.textContent = name;
    const t2 = svg("text", { class: "pd", x: lx, y: 111, "text-anchor": anchor }, g); t2.textContent = prod;
    const ux = side === "left" ? lx - 44 : lx;
    svg("line", { class: "tagline", x1: ux, x2: ux + 44, y1: 118, y2: 118 }, g);
    const go = () => trSetAxis(TR.axis === axis ? "all" : axis);
    g.addEventListener("click", go);
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  };
  spire("openai", 146, "OpenAI", "ChatGPT · Codex", "left");
  spire("anthropic", 214, "Anthropic", "Claude · Claude Code", "right");
  svg("rect", { class: "roof", x: 0, y: H - 4, width: W, height: 4 }, root);
  return root;
}

/* ------------------------------------------------------------- process flow */
const TR_FLOW = { NW: 112, NH: 70, X0: 16, DX: 140, Y0: 30, DY: 104 };
function trBox(s) {
  const { NW, NH, X0, DX, Y0, DY } = TR_FLOW, [c, r] = s.at;
  const x = X0 + c * DX, y = Y0 + r * DY;
  return { x, y, w: NW, h: NH, cx: x + NW / 2, cy: y + NH / 2, r: x + NW, b: y + NH };
}
/* Edges follow the supply lines: same row and next column → straight; another row → along the
   source's row into the gap before the target, then up or down into the target's left side;
   the cell right below → straight up. An edge whose run would cross another stop is left to the
   visit report's "받는 것 / 넘기는 것" links instead of being drawn. */
function trEdge(a, b) {
  const A = trBox(a), B = trBox(b), [ac, ar] = a.at, [bc, br] = b.at;
  if (bc === ac && ar === br + 1) return `M${B.cx},${A.y} V${B.b}`;
  if (bc <= ac) return null;
  for (let c = ac + 1; c < bc; c++) if (TR.idx.cells[`${c},${ar}`]) return null;
  if (ar === br) return bc === ac + 1 ? `M${A.r},${A.cy} H${B.x}` : null;
  const gx = B.x - (TR_FLOW.DX - TR_FLOW.NW) / 2, k = 8, dir = br > ar ? 1 : -1;
  return `M${A.r},${A.cy} H${gx - k} Q${gx},${A.cy} ${gx},${A.cy + dir * k} V${B.cy - dir * k} Q${gx},${B.cy} ${gx + k},${B.cy} H${B.x}`;
}
function trRenderFlow() {
  const stops = TR.data.map.stops, lensMap = trLensMap(), { NW, NH } = TR_FLOW;
  const sums = Object.fromEntries(stops.map((s) => [s.id, trSum(trStopCos(s), lensMap)]));
  const maxSum = Math.max(1e-9, ...Object.values(sums));
  const cols = Math.max(...stops.map((s) => s.at[0])) + 1, rows = Math.max(...stops.map((s) => s.at[1])) + 1;
  const W = TR_FLOW.X0 * 2 + (cols - 1) * TR_FLOW.DX + NW, H = TR_FLOW.Y0 + (rows - 1) * TR_FLOW.DY + NH + 30;
  const root = svg("svg", { class: "tr-flow-svg", viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "AI 공급망 한눈에 보기" });
  const marker = svg("marker", { id: "trArrow", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, svg("defs", {}, root));
  svg("path", { class: "tr-arrowhead", d: "M0,0 L8,4 L0,8 Z" }, marker);
  const here = TR.idx.stopById[TR.stop];
  const edges = svg("g", { class: "tr-edges" }, root);
  for (const s of stops) for (const to of s.to) {
    const b = TR.idx.stopById[to], d = trEdge(s, b);
    if (!d) continue;
    const touch = s === here || b === here;
    const cls = ["tr-edge", s.at[1] === 0 ? "power" : "", touch ? "touch" : ""].filter(Boolean).join(" ");
    svg("path", { class: cls, d, "marker-end": "url(#trArrow)" }, edges);
    svg("path", { class: cls + " run", d }, edges);
  }
  const last = stops[stops.length - 1], LB = trBox(last);
  svg("path", { class: "tr-edge", d: `M${LB.cx},${LB.b} V${LB.b + 16}`, "marker-end": "url(#trArrow)" }, edges);
  const end = svg("text", { class: "tr-end", x: LB.cx, y: LB.b + 28, "text-anchor": "middle" }, root);
  end.textContent = "고객(우리)";
  for (const s of stops) {
    const B = trBox(s), cos = trStopCos(s), sum = sums[s.id];
    const oa = cos.filter((t) => trAxes(t).has("openai")).length, an = cos.filter((t) => trAxes(t).has("anthropic")).length;
    const match = TR.axis === "all" ? null : cos.some((t) => trAxisOn(t));
    const cls = ["tr-node", s.id === TR.stop ? "here" : "", TR.visited.has(s.id) ? "visited" : "", match === false ? "dim" : "",
                 s.id === "labs" ? "product" : ""].filter(Boolean).join(" ");
    const g = svg("g", { class: cls, transform: `translate(${B.x},${B.y})`, tabindex: "0", role: "button", "data-stop": s.id,
                         "aria-label": `${s.no}번째 방문지 ${s.name} — ${s.makes}`, "aria-current": s.id === TR.stop ? "step" : null }, root);
    svg("rect", { class: "box", width: NW, height: NH, rx: 10 }, g);
    svg("circle", { class: "num", cx: 14, cy: 15, r: 9.5 }, g);
    const t = (x, y, str, c, anchor) => { const n = svg("text", { x, y, class: c, "text-anchor": anchor || "start" }, g); n.textContent = str; return n; };
    t(14, 18.5, s.no, "numt", "middle");
    t(29, 20, s.name, "nm");
    t(10, 40, s.makes, "mk");
    if (oa) { svg("circle", { class: "pip oa", cx: 13, cy: 52, r: 3.6 }, g); t(19, 55.5, oa, "pipt"); }
    if (an) { svg("circle", { class: "pip an", cx: oa ? 36 : 13, cy: 52, r: 3.6 }, g); t(oa ? 42 : 19, 55.5, an, "pipt"); }
    if (TR.lens !== "none") {
      t(NW - 8, 55.5, sum ? trSumText(sum) : "—", sum ? "val" : "val zero", "end");
      svg("rect", { class: "gauge-bg", x: 8, y: NH - 8, width: NW - 16, height: 3.5, rx: 1.75 }, g);
      if (sum) svg("rect", { class: "gauge", x: 8, y: NH - 8, width: Math.max(3, (NW - 16) * (sum / maxSum)), height: 3.5, rx: 1.75 }, g);
    }
    if (TR.visited.has(s.id) && s.id !== TR.stop) t(NW - 9, 19, "✓", "seen", "end");
    if (s.no === 1) t(NW / 2, -8, "출발 · 기초 단계", "tag", "middle");
    if (s.id === "labs") t(NW / 2, -8, "AI 모델 완성", "tag product", "middle");
  }
  const lensNote = TR.lens === "none" ? "보유 표시 꺼짐" : TR.lens === "momentum"
    ? "오른쪽 아래 숫자·노란 막대 = 수익률 상위 15종목 중 해당 단계의 종목 수"
    : `오른쪽 아래 숫자·노란 막대 = 해당 단계의 ${TR_LENS_KO[TR.lens]} 합계`;
  mount($("#tr-flow"),
    h("header", null,
      h("div", null, h("h2", null, "AI 공급망 한눈에 보기"),
        h("span", { class: "sub" }, "전력·소재·장비부터 AI 모델·서비스까지 이어지는 공급망 · 상자를 누르면 해당 단계를 살펴봅니다")),
      h("div", { class: "tr-progress", title: "한 번 열어 본 방문지에 ✓가 찍힙니다(이 브라우저에만 저장)" },
        `방문 ${TR.visited.size}/${stops.length}`, h("span", { class: "bar" }, h("i", { style: `width:${(TR.visited.size / stops.length) * 100}%` })))),
    h("div", { class: "tr-flow-scroll" }, root),
    h("p", { class: "tr-legend" },
      "번호 = 방문 순서(기초 1~3 → 칩 제조 4~6 → 서버 구성 7~10 → 데이터센터·클라우드 11~12 → 모델·서비스 13~14) · 노란 선 = 전력 공급 · ",
      h("i", { class: "ax oa" }), "OpenAI·", h("i", { class: "ax an" }), `Anthropic과 이어진 회사 수 · ${lensNote}`));
}

/* ------------------------------------------------------------- leads at a glance */
function trRenderLeads() {
  const lens = trLensMap();
  mount($("#tr-leads"),
    h("header", null,
      h("div", null, h("h2", null, "분야별 대표 기업"),
        h("span", { class: "sub" }, "각 분야의 선도 기업과 선정 근거 · 회사를 누르면 기업 소개, 방문지 이름을 누르면 해당 단계로 이동합니다"))),
    h("div", { class: "tr-leads-grid" }, TR.data.map.stops.map((s) => h("section", { class: "tr-lead-stop" + (s.id === TR.stop ? " here" : "") },
      h("button", { type: "button", class: "tr-lead-h", onclick: () => trSelectStop(s.id, true) },
        h("b", null, `${s.no}. ${s.name}`), h("span", null, s.makes)),
      h("ul", null, s.zones.map((z) => z.special === "axes"
        ? h("li", { class: "axes" },
          h("span", { class: "zn" }, "주요 AI 모델 개발사"),
          h("button", { type: "button", class: "lead-chip axes", onclick: () => trSelectZone(z.id),
                        title: "비상장 기업입니다. 두 회사와 계약·투자 관계가 있는 상장사를 함께 살펴볼 수 있습니다" },
            h("i", { class: "ax oa" }), "OpenAI", h("i", { class: "ax an" }), "Anthropic"))
        : z.lead ? h("li", null,
          h("span", { class: "zn" }, z.name),
          h("button", {
            type: "button", title: z.lead_why,
            class: "lead-chip" + (/[가-힣]/.test(trShort(z.lead)) ? " kr" : "") + (TR.lens !== "none" && z.lead in lens ? " lit" : "")
              + (trAxisOn(z.lead) === false ? " dim" : ""),
            onclick: () => trSelectZone(z.id, z.lead),
          }, `★ ${trShort(z.lead)}`, trSub(z.lead) ? h("small", null, trSub(z.lead)) : null)) : null))))),
    h("p", { class: "tr-legend" },
      "★ = 해당 분야에서 선정한 대표 기업. 점유율·매출·사업 대표성을 바탕으로 골랐으며, 숫자가 있는 근거에는 출처를 표시합니다. 이름에 마우스를 올리면 선정 근거, 누르면 기업 소개와 출처가 열립니다 · ",
      h("span", { class: "lead-chip lit sample" }, "노란 표시"), " = 선택한 표시 대상에 포함된 종목"));
}
function trLeadRow(z) {  // the 대장 line inside a department, with the source that backs its reason
  const mem = (z.members || []).find((x) => x.t === z.lead) || {}, src = z.lead_src || mem.src;
  const sub = trSub(z.lead) || (trCo(z.lead).ko !== trShort(z.lead) ? trCo(z.lead).ko : "");
  return [h("dt", null, "대표 기업"), h("dd", { class: "lead" },
    h("button", { type: "button", class: "linkbtn leadbtn", onclick: () => trSelectZone(z.id, z.lead) }, `★ ${trShort(z.lead)}`),
    sub ? h("span", { class: "muted" }, sub) : null, ` · ${z.lead_why}`, src ? trSrc(src) : null)];
}
const trLeadFirst = (z) => [...(z.members || [])].sort((a, b) => (b.t === z.lead) - (a.t === z.lead));

const trHop = (id) => {
  const s = TR.idx.stopById[id];
  return h("button", { type: "button", class: "tr-hop", onclick: () => trSelectStop(id, true) }, `${s.no}. ${s.name}`);
};

/* ------------------------------------------------------------------ visit report */
function trRenderVisit() {
  const m = TR.data.map, s = TR.idx.stopById[TR.stop];
  const hit = TR.idx.zoneById[TR.zone];
  const z = hit && hit.s === s ? hit.z : s.zones[0];
  TR.zone = z.id;
  const cos = trStopCos(s), at = m.stops.indexOf(s), prev = m.stops[at - 1], next = m.stops[at + 1];
  const stat = (label, lens) => {
    const lm = trLensMap(lens), v = cos.reduce((a, t) => a + (lm[t] || 0), 0), k = cos.filter((t) => t in lm).length;
    return h("span", { class: "chip", title: `${label} 중 이 방문지가 대표인 회사 ${k}곳` }, label, " ", h("b", null, k ? pctAbs(v, 1) : "0%"));
  };
  const oa = cos.filter((t) => trAxes(t).has("openai")).length, an = cos.filter((t) => trAxes(t).has("anthropic")).length;
  const dot = (cls) => h("i", { class: `ax ${cls}`, style: "width:8px;height:8px;border-radius:50%;display:inline-block" });
  const nav = (where) => h("div", { class: `tr-step ${where}` },
    prev ? h("button", { class: "btn", onclick: () => trSelectStop(prev.id, true) }, `← ${prev.no}. ${prev.name}`) : h("span"),
    next ? h("button", { class: "btn primary", onclick: () => trSelectStop(next.id, true) }, `다음 방문지 ${next.no}. ${next.name} →`)
      : h("span", { class: "tr-done" }, "탐방 완료 · AI 서비스가 고객에게 전달되는 단계입니다"));
  mount($("#tr-visit"),
    h("header", { class: "tr-vhead" },
      h("div", { class: "tr-vtitle" },
        h("div", { class: "fl" }, `${s.no}번째 방문지 · ${s.leg}`),
        h("h2", null, s.name, h("span", null, s.site))),
      nav("top")),
    h("div", { class: "tr-io" },
      h("div", { class: "cell in" }, h("span", { class: "k" }, "받는 것"), h("p", null, s.gets),
        h("div", { class: "hops" }, s.from.length ? s.from.map(trHop) : h("span", { class: "tr-hop start" }, "출발점"))),
      h("div", { class: "arrow", "aria-hidden": "true" }, "→"),
      h("div", { class: "cell make" }, h("span", { class: "k" }, "여기서 만드는 것"), h("p", null, s.makes)),
      h("div", { class: "arrow", "aria-hidden": "true" }, "→"),
      h("div", { class: "cell out" }, h("span", { class: "k" }, "다음 단계에 공급하는 것"), h("p", null, s.gives),
        h("div", { class: "hops" }, s.to.length ? s.to.map(trHop) : h("span", { class: "tr-hop start" }, "고객")))),
    h("p", { class: "tr-need" }, s.need),
    h("div", { class: "tr-fstats" },
      stat("TIME", "time"), stat("KoAct", "koact"),
      typeof S !== "undefined" && S ? [stat("7종목", "seven"), stat("내 보유", "mine")] : null,
      h("span", { class: "chip" }, dot("oa"), `OpenAI ${oa}곳`), h("span", { class: "chip" }, dot("an"), `Anthropic ${an}곳`)),
    trSceneFor(s),
    h("h3", { class: "tr-meet" }, "이 단계의 주요 기업", h("span", null, `${s.zones.length}개 분야 · 분야를 누르면 관련 기업을 보여 줍니다`)),
    h("div", { class: "tr-rooms", role: "group", "aria-label": `${s.name}의 사업 분야` }, s.zones.map((zz, k) => trRoom(s, zz, k))),
    trZone(s, z),
    nav("bottom"));
  if (TR.focus) {
    const card = $(`#tr-visit .co[data-t="${CSS.escape(TR.focus)}"]`);
    if (card) requestAnimationFrame(() => card.scrollIntoView({ behavior: "smooth", block: "nearest" }));
  }
}

function trRoom(s, z, k) {
  const lensMap = trLensMap(), vals = Object.values(lensMap), maxW = vals.length ? Math.max(...vals) : 0;
  return h("button", { class: "room", "aria-pressed": String(z.id === TR.zone), "data-zone": z.id },
    h("span", { class: "rn" }, `${s.no}-${k + 1}`, z.bottleneck ? h("span", { class: "bn" }, "병목") : null),
    h("b", null, z.name),
    z.special === "axes" ? h("span", { class: "muted", style: "font-size:11px" }, "비상장 두 회사 · 기업가치·컴퓨팅 확보 계획")
      : h("span", { class: "rm" }, trLeadFirst(z).map((mem) => {
        const chip = trWin(mem.t, lensMap, maxW);
        if (mem.t === z.lead) { chip.classList.add("lead"); chip.firstChild.prepend("★ "); }
        return chip;
      })));
}

function trZone(s, z) {
  if (z.special === "axes") return trAxesRoom(z);
  return h("section", { class: "tr-zone" },
    h("header", null, h("h3", null, z.name), z.bottleneck ? h("span", { class: "tag bn" }, "병목") : null,
      h("span", { class: "muted", style: "font-size:12px" }, `${z.members.length}곳 · ${s.no}. ${s.name}`)),
    h("dl", { class: "zfacts" },
      z.lead ? trLeadRow(z) : null,
      h("dt", null, "필요한 것"), h("dd", null, z.need),
      z.race ? [h("dt", null, "경쟁 구도"), h("dd", null, z.race, z.src && !z.bottleneck ? trSrc(z.src) : null)] : null,
      z.bottleneck ? [h("dt", null, "병목인 이유"), h("dd", null, z.bottleneck, z.src ? trSrc(z.src) : null)] : null),
    h("div", { class: "cos" }, trLeadFirst(z).map((mem, i) => trCard(mem, z, i))));
}

const trSrc = (href) => h("a", { class: "src", href, target: "_blank", rel: "noopener noreferrer" }, "출처");

function trLinkLi(ln) {
  const a = TR_AX[ln.axis];
  const size = [ln.gw ? `${ln.gw}GW` : null, ln.usd_bn ? trUsd(ln.usd_bn) : null].filter(Boolean).join(" · ");
  return h("li", null, h("i", { class: `ax ${a.cls}` }),
    h("span", null, h("b", null, a.ko), " ", ln.what, " ",
      h("span", { class: "meta" }, [ln.date.slice(0, 7), size].filter(Boolean).join(" · ")), " ",
      h("span", { class: `st s-${ln.status}` }, TR_STATUS[ln.status] || ln.status),
      ln.update ? h("span", { class: "upd" }, ` → ${ln.update}`) : null, " ",
      h("a", { href: ln.src, target: "_blank", rel: "noopener noreferrer" }, "출처")));
}

function trCard(mem, z, i) {
  const t = mem.t, c = trCo(t), d = TR.data, links = TR.idx.links[t] || [];
  const w = (k) => ((d.funds[k] || {}).weights || {})[t];
  const lens = trLensMap(), held = TR.lens !== "none" && t in lens;
  const seven = trLensMap("seven")[t], mine = trLensMap("mine")[t], rank = trRank(t);
  const others = (TR.idx.spots[t] || []).filter((x) => x.z !== z);
  const sector = d.sectors[t];
  const hv = (label, v, lensKey) => (v == null ? null
    : h("span", { class: "h" + (TR.lens === lensKey ? " lamp" : "") }, label, " ", h("b", null, pctAbs(v, 1))));
  const mk = { US: null, private: "비상장", fund: "펀드", KR: "한국", JP: "일본", HK: "홍콩", CN: "중국", TW: "대만", EU: "유럽", IN: "인도" }[c.market];
  return h("article", {
    class: "co" + (TR.focus === t ? " focus" : "") + (held ? " held" : ""), "data-t": t,
    style: `animation-delay:${Math.min(i, 12) * 28}ms`,
  },
    h("header", null,
      h("span", { class: "tk" + (/[가-힣]/.test(trShort(t)) ? " kr" : "") }, c.market === "US" ? t : trShort(t)),
      h("span", { class: "nm" }, c.ko !== trShort(t) ? c.ko : c.name),
      h("span", { class: "badges" },
        z.lead === t ? h("span", { class: "tag lead", title: z.lead_why }, "★ 대표") : null,
        mem.pos && (z.lead !== t || mem.pos === "독점") ? h("span", { class: `tag pos-${mem.pos}` }, mem.pos) : null,
        mk ? h("span", { class: "tag" + (c.market === "private" ? " priv" : "") }, mk) : null)),
    h("p", { class: "role" }, mem.role, mem.src ? trSrc(mem.src) : null),
    links.length ? h("ul", { class: "lk" }, links.map(trLinkLi)) : null,
    h("footer", null,
      hv("TIME", w("TIME"), "time"), hv("KoAct", w("KoAct"), "koact"),
      hv("7종목", seven, "seven"), hv("내 보유", mine, "mine"),
      rank ? h("span", { class: "h" + (TR.lens === "momentum" ? " lamp" : "") }, "120일 수익률 ", h("b", null, `${rank.rank}위`)) : null,
      sector && sector !== "미분류" ? h("span", null, sector) : null,
      (d.universe || {})[t] ? h("span", { title: `AI 분석 대상 · ${d.universe[t]}` }, "AI 분석 대상") : null,
      w("TIME") == null && w("KoAct") == null && c.market !== "private" && c.market !== "fund" ? h("span", null, "두 펀드 미보유") : null),
    others.length ? h("div", { class: "also" }, "다른 단계에서의 역할:",
      others.map((x) => h("button", { class: "linkbtn", onclick: () => trGoto(x.z.id, t) }, `${x.s.no}. ${x.s.name} · ${x.z.name}`))) : null);
}

function trAxesRoom(z) {
  const m = TR.data.map;
  const one = (axis) => {
    const a = m.axes[axis], x = TR_AX[axis], links = m.links.filter((l) => l.axis === axis);
    const partners = new Set(links.map((l) => l.t));
    return h("article", { class: "co", style: `border-top:3px solid var(--ax-${x.cls})` },
      h("header", null, h("span", { class: "tk" }, a.name), h("span", { class: "nm" }, a.products), h("span", { class: "badges" }, h("span", { class: "tag priv" }, "비상장"))),
      h("p", { class: "role" }, a.summary),
      h("dl", { class: "profile" }, a.facts.map((fct) => [h("dt", null, fct.label),
        h("dd", null, fct.value, h("span", { class: "meta" }, ` ${fct.date.slice(0, 7)} `),
          h("a", { href: fct.src, target: "_blank", rel: "noopener noreferrer", class: "meta" }, "출처"))])),
      h("footer", null, h("span", null, `계약·투자 상대 ${partners.size}곳 · 기록 ${links.length}건`),
        h("button", { class: "linkbtn", onclick: () => { TR.ledgerAxis = axis; trRenderAxes(); $("#tr-axes").scrollIntoView({ behavior: "smooth" }); } }, "계약·투자 내역 보기")));
  };
  return h("section", { class: "tr-zone" },
    h("header", null, h("h3", null, z.name), h("span", { class: "muted", style: "font-size:12px" }, "두 회사는 비상장입니다 · 관련 공급망 기업과의 계약·투자 관계를 살펴보세요")),
    h("dl", { class: "zfacts" }, h("dt", null, "필요한 것"), h("dd", null, z.need), z.race ? [h("dt", null, "경쟁 구도"), h("dd", null, z.race)] : null),
    h("div", { class: "axes-pair" }, one("openai"), one("anthropic")));
}

/* -------------------------------------------------------------- selection */
function trVisited(id) {
  TR.visited.add(id);
  trStore.set("stop", id);
  trStore.set("visited", JSON.stringify([...TR.visited]));
}
function trSelectStop(id, scroll = false) {
  if (!TR.idx.stopById[id]) return;
  TR.stop = id; TR.zone = null; TR.focus = null; trVisited(id);
  trRenderFlow(); trRenderLeads(); trRenderVisit();
  if (scroll) $("#tr-visit").scrollIntoView({ behavior: "smooth", block: "start" });
}
function trSelectZone(zid, focus = null) {
  const hit = TR.idx.zoneById[zid];
  if (!hit) return;
  TR.stop = hit.s.id; TR.zone = zid; TR.focus = focus; trVisited(hit.s.id);
  trRenderFlow(); trRenderLeads(); trRenderVisit();
}
function trGoto(zid, t) { trSelectZone(zid, t); }
function trFocus(t) {
  const sp = TR.idx.main[t];
  if (!sp) { toast(`${t}는 탐방 경로에 포함되지 않은 종목입니다. ‘보유 종목의 사업 역할’에서 이유를 확인하세요.`); $("#tr-mine").scrollIntoView({ behavior: "smooth" }); return; }
  trSelectZone(sp.z.id, t);
  $("#tr-visit").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ------------------------------------------------------------------ scenes */
/* The cutaways (단면도) are rendered in Blender (desk/scenes/blender/) and arrive with /api/tour
   as `scenes`: each panel is an image plus, per zone, an anchor on the zone's visible pixels, a
   hover mask and a label box placed by desk/scenes/layout.py. Labels print the room number,
   name, ★ lead and 병목 from the map, so they match the room list below. A scene with several
   panels (campus → optics, rack → tray) shows one at a time: the one with most of this stop's
   zones, unless the selected zone is drawn only in another. Stop 13 keeps its drawn spires. */
const TR_CUSTOM = {
  lab: { title: "AI 모델과 서비스가 만들어지는 곳",
         caption: "OpenAI(ChatGPT)와 Anthropic(Claude) · 회사를 누르면 연결된 기업이 흐름도와 기업 소개에서 강조됩니다",
         draw: () => h("div", { class: "lab" }, trSpires()) },
};
const TR_INK = { power: "#f0b429", cold: "#3b8fe0", hot: "#e8603c", fiber: "#25bfe0", light: "#c690ff" };  // colours baked into the renders
const TR_IMG = {};   // panel id -> <img>, reused across re-renders so a click never reloads the picture
TR.pick = null;      // { scene, panel, zone }: a panel tab chosen while that zone was selected

function trRoomText(zid) {  // the same strings desk/scenes/layout.py measured (label_text)
  const { s, z } = TR.idx.zoneById[zid];
  const sub = [z.lead ? `★ ${trShort(z.lead)}` : "", z.bottleneck ? "병목" : ""].filter(Boolean);
  return { num: `${s.no}-${s.zones.indexOf(z) + 1}`, name: z.name, sub, s, z };
}

function trSpotClass(zid) {
  const { s, z } = TR.idx.zoneById[zid], cls = [];
  if (s.id === TR.stop) cls.push("here");
  if (zid === TR.zone) cls.push("on");
  if (TR.axis !== "all") {
    const ok = TR.axis === "both" ? (z.members || []).some((m) => trAxisOn(m.t)) : trZoneAxes(z).has(TR.axis);
    if (!ok) cls.push("away");
  }
  return cls.join(" ");
}

function trSceneFor(s) {
  const all = TR.data.scenes || {}, sc = s.scene && (all.scenes || {})[s.scene];
  if (sc && sc.panels.length) return trRendered(s, s.scene, sc, all.panels);
  const c = s.scene && TR_CUSTOM[s.scene];
  return c ? [h("div", { class: "scene-cap" }, h("b", null, c.title), c.caption), h("div", { class: "scene custom" }, c.draw())] : null;
}

function trRendered(s, sid, sc, P) {
  const panels = sc.panels.map((id) => ({ id, ...P[id] }));
  const known = (sp) => TR.idx.zoneById[sp.zone];
  const mine = (p) => p.spots.filter((sp) => known(sp) && known(sp).s === s).length;
  const has = (p, zid) => p.spots.some((sp) => sp.zone === zid);
  const share = (p) => mine(p) / Math.max(1, p.spots.length);       // ties go to the more focused picture
  const best = panels.reduce((a, b) => (mine(b) > mine(a) || (mine(b) === mine(a) && share(b) > share(a)) ? b : a));
  let cur = TR.pick && TR.pick.scene === sid && TR.pick.zone === TR.zone ? panels.find((p) => p.id === TR.pick.panel) : null;
  if (!cur) cur = TR.zone && !has(best, TR.zone) ? panels.find((p) => has(p, TR.zone)) || best : best;
  const tabs = panels.length > 1 ? h("div", { class: "seg scp-tabs", role: "tablist", "aria-label": `${sc.title} 그림 고르기` },
    panels.map((p, i) => h("button", {
      role: "tab", "aria-selected": String(p === cur),
      onclick: () => { TR.pick = { scene: sid, panel: p.id, zone: TR.zone }; trRenderVisit(); },
    }, `${i + 1}. ${p.title}`, mine(p) ? h("small", null, ` · 이 방문지 ${mine(p)}곳`) : null))) : null;
  const legend = cur.legend.length ? h("div", { class: "scp-legend" }, cur.legend.map(([k, label]) =>
    h("span", null, h("i", { style: `background:${TR_INK[k]}` }), label))) : null;
  const host = h("div", { class: "scene scp-host" }, trPanel(cur));
  trWireScene(host);
  return [
    h("div", { class: "scene-cap" }, h("b", null, sc.title), cur.caption),
    h("div", { class: "scp-bar" }, tabs, legend,
      h("span", { class: "scp-how" }, "이름표나 그림을 누르면 그 분야의 기업이 아래에 열립니다 · 금색 = 이 방문지의 분야 · 점 = OpenAI(파랑)·Anthropic(클레이색)과 연결된 기업")),
    host,
  ];
}

function trPanel(p) {
  const fig = h("figure", { class: "scp", "data-panel": p.id });
  const img = TR_IMG[p.id] ||= h("img", { src: `/static/scenes/${p.img}`, width: p.w, height: p.h, alt: "", decoding: "async", draggable: "false" });
  img.alt = p.title || "단면도";
  const ov = svg("svg", { class: "scp-ov", viewBox: `0 0 ${p.w} ${p.h}`, role: "group", "aria-label": `${p.title || "단면도"} 분야 이름표` });
  const masks = svg("g", { class: "scp-masks" }, ov), lines = svg("g", null, ov), labs = svg("g", null, ov);
  for (const sp of p.spots.filter((x) => TR.idx.zoneById[x.zone])) {
    const z = sp.zone, r = trRoomText(z), cls = trSpotClass(z), ax = trZoneAxes(r.z);
    const [px, py] = sp.anchor, [ex, ey] = sp.end, [x, y, w, hh] = sp.box;
    svg("path", { class: `scp-mask ${cls}`, d: sp.path, "data-zone": z }, masks);
    svg("line", { class: "scp-ldr-u", x1: px, y1: py, x2: ex, y2: ey }, lines);
    svg("line", { class: `scp-ldr ${cls}`, x1: px, y1: py, x2: ex, y2: ey, "data-zone": z }, lines);
    svg("circle", { class: `scp-pin ${cls}`, cx: px, cy: py, r: 4, "data-zone": z }, lines);
    const g = svg("g", { class: `scp-lab ${cls}`, transform: `translate(${x},${y})`, "data-zone": z, "data-n": r.num,
                         "data-x": x, "data-left": String(px > x + w / 2), tabindex: "0", role: "button",
                         "aria-label": `${r.num} ${r.name}${r.sub.length ? ", " + r.sub.join(", ") : ""} — ${r.s.no}번째 방문지 ${r.s.name}` }, labs);
    svg("rect", { width: w, height: hh, rx: 6 }, g);
    const t1 = svg("text", { x: 7, y: 17 }, g);
    svg("tspan", { class: "n" }, t1).textContent = r.num;
    svg("tspan", { class: "nm", dx: 5 }, t1).textContent = r.name;
    if (r.sub.length) {
      const t2 = svg("text", { class: "sub", x: 7, y: 31 }, g);
      r.sub.forEach((part, i) => { svg("tspan", { class: part === "병목" ? "bn" : "" }, t2).textContent = (i ? " · " : "") + part; });
    }
    [...ax].sort().reverse().forEach((a, i) => svg("circle", { class: `ax ${TR_AX[a].cls}`, cx: w - 5 - i * 9, cy: 0, r: 3.6 }, g));
  }
  fig.append(img, ov);
  requestAnimationFrame(() => trFitLabels(ov));
  return fig;
}

/* The layout measured text with a width table; once the page font is known, widen any label
   whose text runs longer, growing away from its leader. */
function trFitLabels(ov) {
  if (!ov.isConnected) return;
  for (const g of ov.querySelectorAll(".scp-lab")) {
    const rect = g.querySelector("rect"), texts = g.querySelectorAll("text");
    let need = 0;
    for (const t of texts) need = Math.max(need, t.getComputedTextLength() + 14);
    const have = +rect.getAttribute("width");
    if (!need || need <= have) continue;
    rect.setAttribute("width", need.toFixed(1));
    for (const c of g.querySelectorAll("circle.ax")) c.setAttribute("cx", +c.getAttribute("cx") + need - have);
    if (g.dataset.left === "true") {
      const t = g.getAttribute("transform").match(/translate\(([-\d.]+),([-\d.]+)\)/);
      g.setAttribute("transform", `translate(${+t[1] - (need - have)},${t[2]})`);
    }
  }
}

function trWireScene(host) {
  const zoneOf = (e) => e.target.closest("[data-zone]");
  const mark = (zid) => {
    for (const n of host.querySelectorAll(".hover")) n.classList.remove("hover");
    if (zid) for (const n of host.querySelectorAll(`[data-zone="${CSS.escape(zid)}"]`)) n.classList.add("hover");
  };
  const tip = (e, g) => {
    const { s, z } = TR.idx.zoneById[g.dataset.zone], lens = trLensMap(), ts = (z.members || []).map((m) => m.t);
    const sum = trSum(ts, lens), r = trRoomText(g.dataset.zone);
    trTipShow(e, h("div", null,
      h("div", { class: "d" }, `${r.num} · ${s.no}번째 방문지 ${s.name}`), h("b", null, z.name),
      h("div", { class: "r" }, ts.slice(0, 7).map(trShort).join(" · ") + (ts.length > 7 ? ` 외 ${ts.length - 7}` : "")),
      TR.lens !== "none" && sum ? h("div", { class: "r" }, `${TR_LENS_KO[TR.lens]} ${trSumText(sum)}`) : null,
      h("div", { class: "r", style: "color:var(--muted)" }, s.id === TR.stop ? "누르면 기업 소개가 열립니다" : "누르면 그 방문지로 이동합니다")));
  };
  host.addEventListener("click", (e) => { const g = zoneOf(e); if (g) trSelectZone(g.dataset.zone); });
  host.addEventListener("keydown", (e) => {
    const g = zoneOf(e);
    if (g && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); trSelectZone(g.dataset.zone); }
  });
  host.addEventListener("mousemove", (e) => {
    const g = zoneOf(e);
    if (!g) { mark(null); return trTipHide(); }
    mark(g.dataset.zone);
    tip(e, g);
  });
  host.addEventListener("mouseleave", () => { mark(null); trTipHide(); });
  host.addEventListener("focusin", (e) => { const g = zoneOf(e); mark(g ? g.dataset.zone : null); });
  host.addEventListener("focusout", () => mark(null));
}

/* ------------------------------------------------------------- two axes */
function trRenderAxes() {
  const m = TR.data.map, lens = trLensMap();
  const partners = { openai: new Set(), anthropic: new Set() };
  for (const l of m.links) partners[l.axis].add(l.t);
  const both = [...partners.openai].filter((t) => partners.anthropic.has(t));
  const only = (a, b) => [...partners[a]].filter((t) => !partners[b].has(t));
  const kindOf = (t, axis) => {
    const ls = (TR.idx.links[t] || []).filter((l) => !axis || l.axis === axis);
    return TR_KINDS.find((k) => ls.some((l) => l.kind === k)) || "유통";
  };
  const chip = (t) => h("button", {
    class: "vchip" + (TR.lens !== "none" && t in lens ? " lit" : ""), onclick: () => trFocus(t),
    title: (TR.idx.links[t] || []).map((l) => `${TR_AX[l.axis].ko}: ${l.what}`).join("\n"),
  }, trShort(t), h("small", null, trSub(t)));
  const grouped = (ts, axis) => TR_KINDS.map((k) => {
    const list = ts.filter((t) => kindOf(t, axis) === k);
    return list.length ? h("div", { class: "vgroup" }, h("span", { class: "eyebrow" }, k), h("div", { class: "vchips" }, list.map(chip))) : null;
  });
  const profile = (axis) => {
    const a = m.axes[axis];
    return h("dl", { class: "profile" }, a.facts.map((f) => [h("dt", null, f.label),
      h("dd", null, f.value, h("span", { class: "meta" }, ` ${f.date.slice(0, 7)}`))]));
  };
  const col = (axis, list) => {
    const a = m.axes[axis], x = TR_AX[axis];
    return h("div", { class: `col ${x.cls}` },
      h("h3", null, h("i", { class: `ax ${x.cls}` }), a.name, h("small", { class: "muted", style: "font-size:12px;font-weight:500" }, a.products)),
      h("div", { class: "sub" }, `지도에서 ${x.ko}에만 연결된 기업 ${list.length}곳`),
      profile(axis), grouped(list, axis));
  };
  const gwBars = (axis) => {
    const rows = m.links.filter((l) => l.axis === axis && l.gw).sort((a, b) => b.gw - a.gw);
    const max = Math.max(1, ...m.links.filter((l) => l.gw).map((l) => l.gw));
    const x = TR_AX[axis];
    return h("div", { class: "gw" },
      h("h3", null, `${x.ko}의 컴퓨팅 확보 계획`), h("div", { class: "sub" }, "AI 연산 자원 확보 규모를 전력 용량(GW)으로 비교 · 빗금 = 의향서·언론 보도 단계"),
      rows.length ? rows.map((l) => h("div", { class: "gw-row", "data-t": l.t, title: `${l.what} (${l.date}, ${TR_STATUS[l.status]})` },
        h("span", { class: "nm" }, h("b", null, trShort(l.t)), l.kind),
        h("span", { class: "track" },
          h("i", { class: `b ${x.cls}${l.status === "의향" || l.status === "보도" ? " soft" : ""}`, style: `width:${(l.gw / max) * 100}%` }),
          h("span", { class: "v" }, `${l.gw}GW`))))
        : h("p", { class: "muted" }, "전력 용량(GW)이 공개된 계약이 없습니다."));
  };
  const rows = m.links
    .filter((l) => (TR.ledgerAxis === "all" || l.axis === TR.ledgerAxis) && (TR.ledgerKind === "all" || l.kind === TR.ledgerKind))
    .sort((a, b) => b.date.localeCompare(a.date))
    .map((l) => h("tr", null,
      h("td", { class: "mono" }, l.date),
      h("td", null, h("i", { class: `ax ${TR_AX[l.axis].cls}` }), TR_AX[l.axis].ko),
      h("td", null, h("button", { class: "linkbtn", style: "text-decoration:none;color:var(--ink);font-family:var(--mono);font-weight:600", onclick: () => trFocus(l.t) }, trShort(l.t)),
        h("span", { class: "muted", style: "font-size:11.5px" }, ` ${trSub(l.t)}`)),
      h("td", null, l.kind),
      h("td", { class: "what" }, l.what, l.update ? h("span", { class: "upd" }, `→ ${l.update}`) : null),
      h("td", { class: "r" }, [l.gw ? `${l.gw}GW` : null, l.usd_bn ? trUsd(l.usd_bn) : null].filter(Boolean).join(" · ") || "—"),
      h("td", null, h("span", { class: `st s-${l.status}` }, TR_STATUS[l.status])),
      h("td", null, h("a", { href: l.src, target: "_blank", rel: "noopener noreferrer" }, "출처"))));
  const seg = (items, cur, set) => h("div", { class: "seg", role: "radiogroup" }, items.map(([k, label]) =>
    h("button", { role: "radio", "aria-checked": String(cur === k), onclick: () => { set(k); trRenderAxes(); } }, label)));
  const kinds = TR_KINDS.filter((k) => m.links.some((l) => l.kind === k));
  mount($("#tr-axes"),
    h("header", null, h("h2", null, "OpenAI·Anthropic 계약과 투자"),
      h("span", { class: "sub" }, "기업별 계약·투자 관계 · 회사 발표와 주요 언론 기준 · 기업 이름을 누르면 탐방 지도에서 해당 기업으로 이동합니다")),
    h("div", { class: "tr-venn" },
      col("openai", only("openai", "anthropic")),
      h("div", { class: "col both" },
        h("h3", null, h("i", { class: "ax", style: "background:linear-gradient(90deg,var(--ax-oa) 50%,var(--ax-an) 50%)" }), "양쪽 모두"),
        h("div", { class: "sub" }, `두 회사 모두와 계약·투자 관계가 기록된 기업 ${both.length}곳`),
        grouped(both, null)),
      col("anthropic", only("anthropic", "openai"))),
    h("div", { class: "tr-gw" }, gwBars("openai"), gwBars("anthropic")),
    h("p", { class: "gw-note" }, "계약별 자원이 중복될 수 있어 합산하지 않습니다. 예를 들어 엔비디아 칩이 오라클·코어위브 데이터센터에 들어갈 수 있습니다. GW는 데이터센터 전력 용량을 나타냅니다."),
    h("div", { class: "ledger-bar" }, h("h3", null, `계약·투자 내역 ${rows.length}건`),
      seg([["all", "전체"], ["openai", "OpenAI"], ["anthropic", "Anthropic"]], TR.ledgerAxis, (k) => { TR.ledgerAxis = k; }),
      seg([["all", "전체 유형"], ...kinds.map((k) => [k, k])], TR.ledgerKind, (k) => { TR.ledgerKind = k; })),
    h("div", { class: "ledger" }, table([["날짜"], ["상대"], ["회사"], ["종류"], ["내용"], ["규모", 1], ["상태"], [""]], rows)));
}

/* --------------------------------------------------- every holding placed on the flow */
function trRenderMine() {
  const d = TR.data, key = ["time", "koact", "seven", "mine"].includes(TR.lens) ? TR.lens : "time";
  const lm = trLensMap(key), outside = Object.fromEntries((d.map.outside || []).map((o) => [o.t, o]));
  const rows = Object.entries(lm).sort((a, b) => (b[1] || 0) - (a[1] || 0));
  const total = rows.reduce((a, [, w]) => a + (w || 0), 0);
  const inFlow = rows.reduce((a, [t, w]) => a + (TR.idx.main[t] ? w || 0 : 0), 0);
  const body = rows.map(([t, w]) => {
    const sp = TR.idx.main[t], c = d.map.companies[t], o = outside[t], ax = trAxes(t);
    const leads = (TR.idx.spots[t] || []).filter((x) => x.z.lead === t).map((x) => x.z.name);
    return h("tr", { class: sp ? "go" : "off", onclick: sp ? () => trFocus(t) : null, title: sp ? "누르면 해당 방문지의 기업 소개로 이동합니다" : (o ? o.why : "") },
      h("td", null, h("b", { class: "mono" }, c && c.market !== "US" ? trShort(t) : t),
        leads.length ? h("span", { class: "leadstar", title: `대표 분야: ${leads.join(", ")}` }, " ★") : null,
        h("span", { class: "muted" }, ` ${c ? trSub(t) || (c.market !== "US" ? "" : c.ko) : (o && o.ko) || ""}`)),
      h("td", { class: "r mono" }, pctAbs(w || 0, 1)),
      h("td", null, sp ? h("span", { class: "tr-stopchip" }, `${sp.s.no}. ${sp.s.name}`) : h("span", { class: "muted" }, o ? "경로 밖" : "배정 전")),
      h("td", null, sp ? sp.z.name : (o && o.group) || ""),
      h("td", { class: "role" }, sp ? sp.mem.role : (o ? o.why : "방문지가 아직 없습니다. ‘AI 산업 지도 갱신’을 요청하면 배정합니다.")),
      h("td", { class: "axes" }, ax.has("openai") ? h("i", { class: "ax oa", title: "OpenAI와 계약·투자" }) : null,
        ax.has("anthropic") ? h("i", { class: "ax an", title: "Anthropic과 계약·투자" }) : null));
  });
  const note = TR.lens === key ? "" : " · 보유 비중 표는 TIME 기준으로 표시합니다";
  mount($("#tr-mine"),
    h("header", null, h("h2", null, "보유 종목의 사업 역할"),
      h("span", { class: "sub" }, `${TR_LENS_KO[key]}이 높은 순서 · 각 종목이 공급망에서 맡는 역할${note}`),
      h("div", { class: "right" },
        h("span", { class: "chip" }, "공정 안 ", h("b", null, pctAbs(inFlow, 1))),
        h("span", { class: "chip" }, "경로 밖 ", h("b", null, pctAbs(Math.max(0, total - inFlow), 1))))),
    rows.length ? h("div", { class: "table-wrap tr-mine-wrap" }, h("table", { class: "tr-mine-t" },
      h("thead", null, h("tr", null, h("th", null, "종목"), h("th", { class: "r" }, "비중"), h("th", null, "방문지"),
        h("th", null, "사업 분야"), h("th", null, "하는 일"), h("th", null, "연결 관계"))),
      h("tbody", null, body)))
      : h("p", { class: "muted" }, key === "mine" ? "기록된 보유 종목이 없습니다. ‘내 계좌·주문’에서 매매 내역을 기록하세요." : "표시할 보유 종목이 없습니다."));
}

function trRenderNotes() {
  const d = TR.data, m = d.map;
  mount($("#tr-notes"),
    h("b", null, "읽는 법"), " — 흐름도의 번호를 따라 전력·소재·장비부터 칩·서버·데이터센터, AI 모델·서비스까지 살펴봅니다. ",
    "각 기업은 주요 역할을 맡는 방문지에 배치하고, 다른 역할은 기업 소개의 ‘다른 단계에서의 역할’로 연결합니다. ",
    "‘병목’은 공급이 수요를 따라가기 어려운 분야, ‘독점·선두·추격’은 경쟁 구도에서의 위치입니다. ★는 분야별 대표 기업이며, 주가 강세를 나타내는 거인의 어깨의 ◆와 구분합니다.", h("br"),
    h("b", null, "자료 출처"), ` — 공급망 배치와 계약·투자 관계는 회사 발표와 주요 언론 보도를 정리한 것입니다. 각 항목의 ‘출처’에서 확인할 수 있습니다. ‘의향서(LOI)’와 ‘언론 보도’는 계약 확정을 뜻하지 않습니다. 지도 기준일 ${m.as_of}. `,
    `보유 비중은 검증한 펀드 PDF(${["TIME", "KoAct"].filter((k) => d.funds[k]).map((k) => `${k} ${d.funds[k].date}`).join(" · ")})를 사용합니다. 7종목·내 보유·수익률 순위는 ‘내 계좌·주문’과 같은 자료입니다. 보유 표시는 매수 추천이 아닙니다.`);
}

/* ----------------------------------------------------------------- tooltip */
function trTipShow(e, content) {
  const tip = $("#tr-tip");
  mount(tip, content);
  tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth, ht = tip.offsetHeight;
  tip.style.left = Math.min(window.innerWidth - w - 8, e.clientX + pad) + "px";
  tip.style.top = (e.clientY + pad + ht > window.innerHeight ? e.clientY - ht - pad : e.clientY + pad) + "px";
}
function trTipHide() { $("#tr-tip").style.opacity = 0; }
function trWinTip(e, t) {
  const c = trCo(t), sp = TR.idx.main[t], lens = trLensMap(), links = TR.idx.links[t] || [];
  const w = (k) => ((TR.data.funds[k] || {}).weights || {})[t];
  const held = ["TIME", "KoAct"].filter((k) => w(k) != null).map((k) => `${k} ${pctAbs(w(k), 1)}`).join(" · ");
  const n = (axis) => links.filter((l) => l.axis === axis).length;
  trTipShow(e, h("div", null,
    h("div", { class: "d" }, `${t}${c.market === "private" ? " · 비상장" : ""}`),
    h("b", null, c.ko), c.ko !== c.name ? h("span", { class: "muted" }, ` ${c.name}`) : null,
    sp ? h("div", { class: "r" }, `${sp.s.no}. ${sp.s.name} · ${sp.z.name} — ${sp.mem.role}`) : null,
    held ? h("div", { class: "r" }, held) : null,
    TR.lens !== "none" && ["seven", "mine"].includes(TR.lens) && t in lens ? h("div", { class: "r" }, `${TR_LENS_KO[TR.lens]} ${pctAbs(lens[t], 1)}`) : null,
    n("openai") ? h("div", { class: "x" }, h("i", { style: "background:var(--ax-oa)" }), `OpenAI ${n("openai")}건`) : null,
    n("anthropic") ? h("div", { class: "x" }, h("i", { style: "background:var(--ax-an)" }), `Anthropic ${n("anthropic")}건`) : null));
}

/* ------------------------------------------------------------------ wiring */
(function trWire() {
  const flow = $("#tr-flow");
  flow.addEventListener("click", (e) => {
    const g = e.target.closest("[data-stop]");
    if (g) trSelectStop(g.dataset.stop, true);
  });
  flow.addEventListener("keydown", (e) => {
    const g = e.target.closest("[data-stop]");
    if (!g) return;
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); trSelectStop(g.dataset.stop, true); return; }
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const stops = TR.data.map.stops, i = stops.findIndex((s) => s.id === g.dataset.stop);
    const next = stops[i + (e.key === "ArrowLeft" ? -1 : 1)];
    if (next) { trSelectStop(next.id); $(`#tr-flow [data-stop="${next.id}"]`).focus(); }
  });
  flow.addEventListener("mousemove", (e) => {
    const g = e.target.closest("[data-stop]");
    if (!g) return trTipHide();
    const s = TR.idx.stopById[g.dataset.stop], lens = trLensMap(), cos = trStopCos(s);
    const held = TR.lens === "none" ? [] : cos.filter((t) => t in lens).sort((a, b) => (lens[b] || 0) - (lens[a] || 0));
    trTipShow(e, h("div", null,
      h("div", { class: "d" }, `${s.no}번째 방문지 · ${s.leg}`), h("b", null, `${s.name} — ${s.site}`),
      h("div", { class: "r" }, `받는 것: ${s.gets} → 만드는 것: ${s.makes} → 공급하는 것: ${s.gives}`),
      h("div", { class: "r" }, "대표 기업: " + s.zones.filter((z) => z.lead).map((z) => `${z.name} ★${trShort(z.lead)}`).join(" · ")),
      held.length ? h("div", { class: "r" }, `${TR_LENS_KO[TR.lens]}: ${held.slice(0, 6).map(trShort).join(" · ")}${held.length > 6 ? ` 외 ${held.length - 6}` : ""}`) : null));
  });
  flow.addEventListener("mouseleave", trTipHide);
  const visit = $("#tr-visit");
  visit.addEventListener("click", (e) => {
    const room = e.target.closest(".room");
    if (!room) return;
    const win = e.target.closest(".win");
    trSelectZone(room.dataset.zone, win ? win.dataset.t : null);
  });
  visit.addEventListener("mousemove", (e) => { const win = e.target.closest(".room .win"); if (win) trWinTip(e, win.dataset.t); else if (!e.target.closest(".scene")) trTipHide(); });
  visit.addEventListener("mouseleave", trTipHide);
  $("#tr-axes").addEventListener("mousemove", (e) => {
    const row = e.target.closest(".gw-row");
    if (!row) return trTipHide();
    trWinTip(e, row.dataset.t);
  });
  $("#tr-axes").addEventListener("mouseleave", trTipHide);
  const find = $("#tr-find");
  find.addEventListener("change", () => trFind(find.value));
  find.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); trFind(find.value); } });
  // app.js's router shows #tour-view and announces it; the map loads on the first visit
  window.addEventListener("desk:tab", (e) => { if (e.detail === "tour") trLoad(); });
  document.addEventListener("desk:state", () => { if (TR.data && !$("#tour-view").hidden) trRenderAll(); });
})();
