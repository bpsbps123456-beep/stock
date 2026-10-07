"use strict";
/* 거인의 어깨 — renders /api/giant (정본 docs/desk_system.md §9). Reuses h()/svg()/$/mount/md/toast
   from app.js. 관찰용 화면이다: 매수 추천·목표 비중·주문 수량이 없다.
   Charts are emphasis scatters (the chosen sector in brass, everything else in the context gray);
   every plotted value is also in the tables. 상승·매수 = red, 하락·매도 = blue, with sign or ▲▼. */

const GI = { data: null, loading: null, lens: "since_low", level: "tour", sel: null,
  sectorGrade: "all", stockGrade: "all", stockScope: "selected",
  sectorQuery: "", stockQuery: "", metricView: "strength", sectorDetails: false,
  sort: null, starsAll: false, ready: false };
const GI_BUY = new Set(["매수", "신규"]);
const GI_SELL = new Set(["매도", "편출"]);
const GI_CHECKS = ["대표 분야 강세", "대표 분야 내 상위권", "전고점 근접", "상승일 거래량 우세"];

const giPct = (v, d = 0) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(d)}%`);
const giDir = (v) => (v == null || v === 0 ? "" : v > 0 ? "up" : "down");
const giNum = (v, d = 1) => (v == null ? "—" : v.toFixed(d));
// 달러 시총을 억·조 단위로: 4.5조 달러, 482억 달러, 9.1억 달러.
const giUsd = (v) => (v == null ? "—" : v >= 1e12 ? `${(v / 1e12).toFixed(1)}조 달러`
  : v >= 1e9 ? `${Math.round(v / 1e8).toLocaleString("ko-KR")}억 달러` : `${(v / 1e8).toFixed(1)}억 달러`);
// 0.005%부터 90%대까지 퍼진 비율을 자릿수만 바꿔 같은 폭으로 보여 준다.
const giSmallPct = (v) => (v == null ? "—"
  : `${v < 0.1 ? v.toFixed(3) : v < 1 ? v.toFixed(2) : v < 10 ? v.toFixed(1) : v.toFixed(0)}%`);
// 회복 등급 칸: 등급 글자 + 회복률 중앙값. 급락을 맞은 종목이 적으면 비운다.
function giGradeCell(r, rules) {
  const tip = r.grade
    ? `급락을 맞은 ${r.recovery_n}종목이 빠진 폭의 ${r.recovery.toFixed(0)}%를 되찾았습니다(중앙값). 등급 하한: ${rules.grades.map(([g, f]) => `${g} ${f}%`).join(" · ")} · 그 아래 D`
    : `급락을 맞은 종목이 ${r.recovery_n ?? 0}개라 등급을 매기지 않습니다(${rules.grade_min}개 이상 필요).`;
  return h("td", { class: "r num", title: tip }, r.grade
    ? [h("span", { class: `gi-grade g-${r.grade}` }, r.grade), ` ${r.recovery.toFixed(0)}%`]
    : h("span", { class: "muted" }, "—"));
}

function giStockGradeCell(s) {
  const tip = s.grade
    ? `이 종목이 급락에서 빠진 폭의 ${s.recovery.toFixed(1)}%를 되찾았습니다. 분야 등급과는 별도로 계산합니다.`
    : "회복 등급 대상이 아닙니다. 자료가 없거나 급락 직전 전고점 근접·낙폭 기준을 충족하지 않았습니다.";
  return h("td", { class: "r num", title: tip }, s.grade
    ? [h("span", { class: `gi-grade g-${s.grade}` }, s.grade), ` ${s.recovery.toFixed(0)}%`]
    : h("span", { class: "muted" }, "—"));
}

function giSegment(label, items, cur, set) {
  return h("div", { class: "gi-ctl" }, h("span", { class: "gi-ctl-label" }, label),
    h("div", { class: "gi-seg", role: "radiogroup", "aria-label": label }, items.map(([k, text, tip]) => h("button", {
      role: "radio", "aria-checked": String(cur === k), title: tip || null, onclick: () => set(k),
    }, text))));
}

const giMatchesGrade = (row, grade) => grade === "all" || (grade === "none" ? row.grade == null : row.grade === grade);
function giGradeOptions(rows) {
  return [["all", `전체 ${rows.length}`], ...["S", "A", "B", "C", "D", "none"].map((g) =>
    [g, `${g === "none" ? "미산정" : g} ${rows.filter((r) => giMatchesGrade(r, g)).length}`])];
}

function giSearch(id, label, placeholder, value, update) {
  const changed = (e) => {
    if (e.isComposing) return;
    const input = e.currentTarget, start = input.selectionStart, end = input.selectionEnd;
    update(input.value);
    // Filters replace their section. Keep typing and the caret in the new search input.
    const next = document.getElementById(id);
    next.focus({ preventScroll: true });
    if (start != null) next.setSelectionRange(start, end);
  };
  return h("label", { class: "gi-ctl gi-search" }, h("span", { class: "gi-ctl-label" }, label),
    h("input", { id, type: "search", value, placeholder, autocomplete: "off", oninput: changed, oncompositionend: changed }));
}

function giLenses(a) {
  return [
    ["since_low", `저점(${md(a.low_date)}) 이후`, "기준 지수의 저점일 이후 수익률입니다. 낙폭이 컸던 종목의 반등이 크게 보일 수 있습니다."],
    ["r1m", "최근 1개월", "최근 21거래일 수익률로 현재 강한 종목과 섹터를 비교합니다."],
    ["cycle", `급락 전(${md(a.peak_date)}) 대비`, "기준 지수가 급락하기 전 고점일의 각 종목 종가와 현재 종가를 비교합니다. 종목 자체의 전고점과는 기준이 다릅니다."],
    ["crash", `급락기(${md(a.peak_date)}→${md(a.low_date)})`, "기준 지수가 고점에서 저점으로 하락한 기간의 수익률입니다. 높을수록 해당 기간에 잘 버틴 종목입니다."],
  ];
}
const giLensLabel = (k) => (giLenses(GI.data.anchor).find(([key]) => key === k) || [k, k])[1];

/* ------------------------------------------------------------------ load */
async function giLoad(force) {
  if (GI.loading) return GI.loading;
  const root = $("#giant-view");
  root.classList.add("gi-busy");
  if (!GI.data) {
    mount(root, h("section", { class: "card gi-wait" },
      h("div", { class: "eyebrow" }, "거인의 어깨"),
      h("p", {}, "AI 분석 대상의 가격·실적·수급을 계산하고 있습니다. 첫 계산에는 약 30~60초가 걸립니다. 이후 60분 동안은 저장된 결과를 보여 줍니다.")));
  }
  GI.loading = fetch("/api/giant" + (force ? "?refresh=1" : ""), { cache: "no-store" })
    .then(async (r) => {
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.error || r.statusText);
      return j;
    })
    .then((d) => {
      GI.data = d;
      if (GI.level === "tour" && !d.tour_sectors?.length) GI.level = "sub";
      giEnsureSelection();
      giRender();
      if (force) toast("거인의 어깨를 다시 계산했습니다");
    })
    .catch((e) => {
      // A desk started before this tab existed serves the new page but not /api/giant.
      const msg = /not found/i.test(e.message)
        ? "거인의 어깨를 열려면 데스크를 재시작하세요. 실행 창을 닫고 ‘야간데스크.cmd’를 다시 실행하면 됩니다."
        : "분석 결과를 계산하지 못했습니다. " + e.message;
      if (GI.data) toast(msg);
      else mount(root, h("section", { class: "card" }, h("p", { class: "err" }, msg)));
    })
    .finally(() => {
      GI.loading = null;
      root.classList.remove("gi-busy");
      const again = $("#gi-controls .btn");
      if (again) again.disabled = false;
    });
  return GI.loading;
}

/* ------------------------------------------------------------- structure */
function giRows() {
  const d = GI.data;
  if (GI.level === "tour") {
    return (d.tour_sectors || []).map((s) => ({ ...s, label: s.sub, parent: s.group }));
  }
  if (GI.level === "group") {
    return d.groups.map((g) => ({ ...g, key: g.group, label: g.group, members: g.subs.flatMap((s) => s.members) }));
  }
  return d.groups.flatMap((g) => g.subs.map((s) => ({ ...s, key: `${g.group} · ${s.sub}`, label: s.sub, parent: g.group })));
}
const giSectorRows = () => giRows().filter((r) => giMatchesGrade(r, GI.sectorGrade)
  && `${r.label} ${r.parent || ""}`.toLocaleLowerCase().includes(GI.sectorQuery.trim().toLocaleLowerCase()));
function giSorted(rows, key = GI.lens) {
  return [...rows].sort((a, b) => (b[key] ?? -1e9) - (a[key] ?? -1e9));
}
function giEnsureSelection() {
  const rows = giSectorRows();
  if (!rows.some((r) => r.key === GI.sel)) GI.sel = giSorted(rows)[0]?.key ?? null;
}
function giSelected() {
  const row = giSectorRows().find((r) => r.key === GI.sel)
    || { label: "해당 분야 없음", key: null, members: [] };
  return { row, stocks: row.members.map((s) => GI.data.stocks[s]).filter(Boolean) };
}
function giStockUniverse() {
  if (GI.stockScope === "all") return Object.values(GI.data.stocks);
  if (GI.stockScope === "fields") {
    return [...new Set(giSectorRows().flatMap((r) => r.members))].map((s) => GI.data.stocks[s]).filter(Boolean);
  }
  return giSelected().stocks;
}
const giStockRows = () => giStockUniverse().filter((s) => giMatchesGrade(s, GI.stockGrade)
  && `${s.name} ${s.symbol}`.toLocaleLowerCase().includes(GI.stockQuery.trim().toLocaleLowerCase()));
function giStockScopeLabel() {
  const scope = GI.stockScope === "all" ? "전체 AI 종목" : GI.stockScope === "fields" ? "표시된 모든 분야" : giSelected().row.label;
  return scope + (GI.stockGrade === "all" ? "" : ` · ${GI.stockGrade === "none" ? "등급 미산정" : GI.stockGrade + "등급 종목"}`);
}
function giSelect(key, scroll) {
  GI.sel = key;
  GI.sort = null;
  giRenderSectors(); giRenderCharts(); giRenderMembers(); giRenderWhale();
  if (scroll) jumpTo("gi-members");
}
function giSelectStock(sym) {
  const s = GI.data.stocks[sym];
  const field = s.tour_fields?.[0];
  GI.level = field ? "tour" : "sub";
  GI.sectorGrade = "all"; GI.stockGrade = "all"; GI.stockScope = "selected";
  GI.sectorQuery = ""; GI.stockQuery = "";
  giRenderControls();
  giSelect(field ? field.key : `${s.group} · ${s.sub}`, true);
}

function giRender() {
  const root = $("#giant-view");
  if (!GI.ready || !$("#gi-sectors", root)) {
    mount(root,
      h("section", { class: "gi-head", id: "gi-head" }),
      sectionNav("거인의 어깨 바로가기", [["gi-kpis", "시장 요약"], ["gi-stars", "주도주 후보"],
        ["gi-sectors", "세부 분야"], ["gi-members", "종목 비교"], ["gi-whale", "펀드·투자자 수급"], ["gi-charts", "비교 차트"]]),
      h("section", { class: "gi-kpis", id: "gi-kpis", "aria-label": "기준 구간 요약" }),
      h("section", { class: "card gi-stars", id: "gi-stars" }),
      h("div", { class: "gi-explorer" },
        h("section", { class: "card gi-sectors", id: "gi-sectors" }),
        h("section", { class: "card gi-members", id: "gi-members" })),
      h("section", { class: "card gi-whale", id: "gi-whale" }),
      h("section", { class: "card gi-charts", id: "gi-charts" }),
      h("details", { class: "notes-panel" }, h("summary", {}, "계산 기준·등급 읽는 법·자료 출처"),
        h("footer", { class: "notes", id: "gi-notes" })));
    GI.ready = true;
  }
  giRenderHead(); giRenderKpis(); giRenderStars(); giRenderSectors(); giRenderCharts();
  giRenderMembers(); giRenderWhale(); giRenderNotes();
}

/* ----------------------------------------------------------------- head */
function giRenderHead() {
  const d = GI.data, a = d.anchor;
  mount($("#gi-head"),
    h("div", { class: "gi-title" },
      h("div", { class: "eyebrow" }, `AI 섹터·주도주 분석 · 시세 기준 ${md(a.last_date)} · 분석 갱신 ${d.computed_at.slice(5, 16).replace("T", " ")} (한국시간)`),
      h("h1", {}, "거인의 어깨 ", h("span", {}, `AI ${d.universe.n}종목의 강세와 수급 흐름`)),
      h("p", { class: "gi-lede" },
        "분야를 고른 뒤 종목의 회복 등급과 수급을 비교하세요. ‘전체 AI 종목’에서는 모든 분야의 S등급 종목을 함께 볼 수 있습니다.")),
    h("div", { class: "gi-controls", id: "gi-controls" }));
  giRenderControls();
}

function giRenderControls() {
  mount($("#gi-controls"),
    giSegment("비교 기간", giLenses(GI.data.anchor), GI.lens, (k) => {
      GI.lens = k; GI.sort = null;
      giRenderControls(); giRenderStars(); giRenderSectors(); giRenderCharts(); giRenderMembers(); giRenderWhale();
    }),
    giSegment("분류 기준", [
      ...(GI.data.tour_sectors?.length ? [["tour", `사업 분야 ${GI.data.tour_sectors.length}`, "AI 산업 지도의 사업 역할로 나눕니다. CPU·GPU처럼 같은 회사가 여러 분야에 속할 수 있습니다."]] : []),
      ["sub", `대표 사업 분류 ${GI.data.groups.reduce((n, g) => n + g.subs.length, 0)}`, "회사별 대표 사업 한 곳으로 분류합니다. ◆ 주도주 후보의 비교 기준입니다."],
      ["group", `산업 대분류 ${GI.data.groups.length}`, "칩·메모리·전력 등 큰 산업 단위로 비교합니다."],
    ], GI.level, (k) => {
      if (k === GI.level) return;
      const members = new Set(giSelected().row.members);
      GI.level = k;
      const rows = giSorted(giSectorRows());
      GI.sel = (rows.find((r) => r.members.some((s) => members.has(s))) || rows[0])?.key ?? null;
      GI.sort = null;
      giRenderControls(); giRenderSectors(); giRenderCharts(); giRenderMembers(); giRenderWhale();
    }),
    h("button", { class: "btn", title: "최신 가격·실적·수급을 불러와 다시 계산합니다(약 30~60초)", onclick: (e) => {
      e.currentTarget.disabled = true; giLoad(true);
    } }, "↻ 분석 갱신"));
}

function giRenderKpis() {
  const d = GI.data, a = d.anchor, stars = Object.values(d.stocks).filter((s) => s.star).length;
  const tile = (label, value, sub, dir) => h("div", { class: "gi-kpi" },
    h("div", { class: "gi-kpi-label" }, label),
    h("div", { class: `gi-kpi-value ${dir || ""}` }, value),
    h("div", { class: "gi-kpi-sub" }, sub));
  mount($("#gi-kpis"),
    tile(`${a.index} 급락기 수익률`, giPct(a.crash, 1), `${md(a.peak_date)} → ${md(a.low_date)}`, giDir(a.crash)),
    tile(`${a.index} 저점 이후 수익률`, giPct(a.since_low, 1), `${md(a.low_date)} → ${md(a.last_date)} · 급락 전 대비 ${giPct(a.vs_peak, 1)}`, giDir(a.since_low)),
    tile("AI 종목 저점 이후 수익률", giPct(d.universe.since_low, 1), `중앙값 · 급락 전 대비 ${giPct(d.universe.cycle, 1)}`, giDir(d.universe.since_low)),
    tile("◆ 주도주 후보", `${stars}종목`, "4개 기준 충족 · 대표 분야 강세·순위·전고점·거래량"),
    giBreadthTile(d.breadth));
}

/* 시장 폭: 주도주 신호가 통하던 장인지 알려 주는 칸. 검증 수치는 giant.py BREADTH_CHECK 하나에서 온다. */
function giBreadthTile(b) {
  if (!b) return null;
  const c = b.check, wide = b.now >= c.split, edge = Math.abs(b.now - c.split) < 3;  // ±3%p는 판단 유보
  const W = 120, H = 28, vals = b.history.map(([, v]) => v);
  const lo = Math.min(...vals, c.split), hi = Math.max(...vals, c.split);
  const X = (i) => (i / Math.max(1, vals.length - 1)) * (W - 4) + 2, Y = (v) => H - 2 - ((v - lo) / (hi - lo || 1)) * (H - 4);
  const spark = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "gi-spark", role: "img",
    "aria-label": `최근 1년 전고점 근접 비율, 지금 ${b.now}%, 기준선 ${c.split}%` });
  svg("line", { x1: 0, x2: W, y1: Y(c.split), y2: Y(c.split), class: "gi-spark-ref" }, spark);
  svg("polyline", { points: vals.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" "), class: "gi-spark-line" }, spark);
  svg("circle", { cx: X(vals.length - 1), cy: Y(vals[vals.length - 1]), r: 3, class: "gi-spark-dot" }, spark);
  return h("div", { class: "gi-kpi gi-breadth", title:
      `${c.period} 주간 재현 검증(${c.dates}개 시점): 시장 폭이 ${c.split}% 이상일 때 ◆ 종목의 20거래일 초과수익은 평균 ${c.wide_excess > 0 ? "+" : ""}${c.wide_excess}%p(적중 ${c.wide_hit}%), 미만일 때 ${c.narrow_excess}%p(적중 ${c.narrow_hit}%)였습니다.` },
    h("div", { class: "gi-kpi-label" }, "전고점 근접 비율"),
    h("div", { class: "gi-kpi-value" }, `${b.now.toFixed(0)}%`, spark),
    h("div", { class: "gi-kpi-sub" }, `AI 종목 중 전고점 -10% 이내 · 1년 중앙값 ${b.median_1y.toFixed(0)}%`),
    h("div", { class: `gi-regime ${edge ? "edge" : wide ? "wide" : "narrow"}` },
      edge ? `≈ 경계 — 기준 ${c.split}%와 3%p 이내라 판단 유보`
        : wide ? `▲ 넓은 장 — 주도주 신호가 통하던 구간(기준 ${c.split}%)` : `▼ 좁은 장 — 주도주 신호가 약하던 구간(기준 ${c.split}%)`));
}

/* ----------------------------------------------------------------- stars */
function giChecks(s) {
  const met = s.checks.filter(Boolean).length;
  const label = `조건 ${met}/4 — ` + GI_CHECKS.map((c, i) => `${c} ${s.checks[i] ? "충족" : "미충족"}`).join(", ");
  return h("span", { class: "gi-checks", role: "img", "aria-label": label, title: label },
    s.checks.map((ok) => h("i", { class: ok ? "on" : "" })));
}

function giRenderStars() {
  const d = GI.data;
  const need = GI.starsAll ? 3 : 4;
  const list = Object.values(d.stocks).filter((s) => s.checks.filter(Boolean).length >= need)
    .sort((a, b) => (b[GI.lens] ?? -1e9) - (a[GI.lens] ?? -1e9));
  mount($("#gi-stars"),
    h("header", {},
      h("h2", {}, "◆ 주도주 후보"),
      h("span", { class: "sub" }, `조건 ${need === 4 ? "4개 모두" : "3개 이상"} 충족 · ${list.length}종목 · ${giLensLabel(GI.lens)} 수익률순`),
      h("div", { class: "right" }, h("button", { class: "linkbtn", onclick: () => { GI.starsAll = !GI.starsAll; giRenderStars(); } },
        GI.starsAll ? "4개 충족 종목만" : "3개 충족 종목도"))),
    list.length ? h("div", { class: "gi-chips" }, list.map((s) => h("button", {
      class: "gi-chip" + (s.star ? " star" : ""), onclick: () => giSelectStock(s.symbol),
      title: `${s.name} — ${s.group} · ${s.sub}`,
    }, giChecks(s), h("b", {}, s.name), h("span", { class: "gi-chip-sub" }, s.sub),
      h("span", { class: `num ${giDir(s[GI.lens])}` }, giPct(s[GI.lens])),
      h("span", { class: "gi-chip-sub" }, `전고점 대비 ${giPct(s.gap_pct)}`))))
      : giHoldNote(d),
    d.anchor.age != null && d.anchor.age <= d.rules.fresh_low_days && list.length
      ? h("p", { class: "gi-hold" }, `≈ 기준 저점이 ${d.anchor.age}거래일 전이라 ‘저점 이후’ 순위가 아직 짧습니다. 지금 후보는 참고만 하세요.`) : null,
    h("p", { class: "gi-foot" }, "선정 조건(대표 사업 분류 기준): ① 분야의 저점 이후 수익률 중앙값이 AI 전체보다 높음 ② 같은 분야에서 저점 이후 수익률 상위 ",
      String(d.rules.leaders_per_sub), "위 이내 ③ 전고점 대비 하락폭 ", String(d.rules.near_high_pct), "% 이내 ④ 최근 ",
      String(d.rules.ud_days), "거래일의 상승일 거래량 합계가 하락일 이상. 2026년 재현 검증 결과는 상단의 ‘전고점 근접 비율’과 아래 설명에 있습니다."));
}

/* ◆가 비는 때(대개 지수가 막 바닥을 만든 때)의 보류 안내. 근거 수치는 giant.py HOLD_CHECK. */
function giHoldNote(d) {
  const c = d.rules.hold_check;
  return h("p", { class: "gi-hold" },
    h("b", {}, "◆ 판단 보류 — "),
    `조건을 모두 채운 종목이 없습니다. 지수가 막 바닥을 만들면 ‘저점 이후’가 모두 0에 가까워져 이렇게 됩니다. `,
    c ? `2026년 검증에서는 이때 AI 전체를 그대로 든 편이 나았습니다(${md(c.date)} 이후 20거래일: AI 전체 +${c.universe}%, 현금 0%, 최근 1개월 강자로 대체 +${c.substitute}%).` : "");
}

/* --------------------------------------------------------------- sectors */
function giHeatStyle(v, base, span) {
  if (v == null || base == null || !span) return null;
  const t = Math.max(-1, Math.min(1, (v - base) / span));
  const mix = Math.round(Math.abs(t) * 30);
  return mix < 3 ? null : `background: color-mix(in srgb, var(${t > 0 ? "--up" : "--down"}) ${mix}%, transparent)`;
}

function giRenderSectors() {
  const d = GI.data, u = d.universe;
  const oldScroll = $("#gi-sectors .gi-scroll");
  const scroll = oldScroll ? [oldScroll.scrollTop, oldScroll.scrollLeft] : [0, 0];
  const rows = giSorted(giSectorRows());
  const cols = [["crash", "급락기"], ["since_low", "저점 이후"], ["r1m", "1개월"], ["cycle", "급락 전 대비"]];
  const span = Object.fromEntries(cols.map(([k]) => [k, Math.max(5, ...rows.map((r) => Math.abs((r[k] ?? u[k]) - u[k])))]));
  mount($("#gi-sectors"),
    h("header", {},
      h("h2", {}, "세부 분야 비교"),
      h("span", { class: "sub" }, `${GI.level === "group" ? "대분류" : "세부 분야"} ${rows.length}/${giRows().length}개 · ${giLensLabel(GI.lens)} 수익률순 · 중앙값 비교`),
      h("div", { class: "right" },
        h("button", { class: "linkbtn", "aria-pressed": String(GI.sectorDetails), onclick: () => {
          GI.sectorDetails = !GI.sectorDetails; giRenderSectors();
        } }, GI.sectorDetails ? "기본 지표 보기" : "전고점·50일선·실적 보기"),
        h("button", { class: "btn primary", onclick: () => {
        GI.stockScope = "all"; GI.stockGrade = "S"; GI.stockQuery = ""; GI.sort = null;
        giRenderMembers(); giRenderCharts(); jumpTo("gi-members");
      } }, "전체 S등급 종목 보기"))),
    h("div", { class: "gi-filters" },
      giSearch("gi-sector-search", "분야 찾기", "CPU, 메모리, 냉각…", GI.sectorQuery, (query) => {
        GI.sectorQuery = query; giEnsureSelection(); GI.sort = null;
        giRenderSectors(); giRenderMembers(); giRenderCharts(); giRenderWhale();
      }),
      giSegment("분야 회복 등급", giGradeOptions(giRows()), GI.sectorGrade, (grade) => {
      GI.sectorGrade = grade; giEnsureSelection(); GI.sort = null;
      giRenderSectors(); giRenderCharts(); giRenderMembers(); giRenderWhale();
    })),
    h("p", { class: "gi-selection" }, giSelected().row.key
      ? [h("span", {}, "선택 분야"), h("b", {}, giSelected().row.label), " · 행을 누르면 종목 목록이 바뀝니다"]
      : "검색어와 등급에 맞는 분야가 없습니다"),
    h("div", { class: "gi-scroll" }, h("table", { class: "gi-table gi-sector-table" + (GI.sectorDetails ? " details" : "") },
      h("thead", {}, h("tr", {},
        h("th", {}, "분야"), h("th", { class: "r", title: "급락 직전 전고점 -15% 안이었고 그 급락에서 15% 넘게 빠진 종목이 빠진 폭 중 되찾은 비율(중앙값)" }, "회복 등급"),
        h("th", { class: "r" }, "종목 수"),
        cols.map(([k, t]) => h("th", { class: "r" + (k === GI.lens ? " lens" : ""), title: giLensLabel(k) }, t)),
        h("th", { class: "r", title: "전고점 대비 하락폭이 10% 이내인 종목의 비율" }, "전고점 근접 비율"),
        h("th", { class: "r", title: "종가가 50일 이동평균보다 높은 종목의 비율" }, "50일선 위 비율"),
        h("th", { class: "r", title: "최근 분기 매출 증가율(전년 대비)의 중앙값" }, "매출 증가율"),
        h("th", { title: "이 분야에서 기준 저점일 이후 주가 수익률이 가장 높은 종목입니다. 위 비교 기간을 바꿔도 이 열의 기준은 저점 이후입니다." }, "저점 이후 1위"))),
      h("tbody", {},
        h("tr", { class: "gi-base" }, h("td", {}, "AI 전체"), d.rules.grades ? giGradeCell(u, d.rules) : h("td", {}),
          h("td", { class: "r" }, String(u.n)),
          cols.map(([k]) => h("td", { class: `r num ${giDir(u[k])}` }, giPct(u[k]))),
          h("td", {}), h("td", {}), h("td", { class: "r num" }, giPct(u.rev_growth)), h("td", {})),
        rows.map((r) => h("tr", {
          class: "gi-row" + (r.key === GI.sel ? " sel" : ""), tabindex: "0", "aria-selected": String(r.key === GI.sel),
          onclick: () => giSelect(r.key, !matchMedia("(min-width: 1700px)").matches),
          onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); giSelect(r.key, !matchMedia("(min-width: 1700px)").matches); } },
        },
          h("td", {}, h("b", {}, r.label), r.parent ? h("span", { class: "gi-parent" }, r.parent) : null),
          d.rules.grades ? giGradeCell(r, d.rules) : h("td", {}),
          h("td", { class: "r num" }, String(r.n)),
          cols.map(([k]) => h("td", { class: `r num ${giDir(r[k])}`, style: giHeatStyle(r[k], u[k], span[k]) }, giPct(r[k]))),
          h("td", { class: "r num" }, r.near_high_pct == null ? "—" : `${r.near_high_pct}%`),
          h("td", { class: "r num" }, r.above50_pct == null ? "—" : `${r.above50_pct}%`),
          h("td", { class: "r num" }, giPct(r.rev_growth)),
          h("td", {}, d.stocks[r.leader] ? d.stocks[r.leader].name : r.leader))),
        !rows.length ? h("tr", {}, h("td", { colspan: "11", class: "gi-empty" }, "검색어와 등급에 맞는 분야가 없습니다. 검색어를 지우거나 다른 등급을 선택하세요.")) : null))),
    h("p", { class: "gi-foot" }, GI.level === "tour"
      ? `AI 산업 지도(${d.tour_as_of})의 사업 역할로 나눴습니다. CPU·GPU처럼 역할이 겹치는 회사는 여러 분야에 표시됩니다. 분야 수익률은 해당 회사들의 주가 중앙값이며, 해당 제품 사업만의 수익률은 아닙니다. `
      : "대표 사업으로 묶은 분류입니다. ",
      `회복 등급은 급락을 맞은 ${d.rules.grade_min}종목 이상일 때 계산합니다. ‘미산정’은 D등급과 다릅니다. 배경색은 AI 전체 중앙값과의 차이입니다.`));
  const nextScroll = $("#gi-sectors .gi-scroll");
  nextScroll.scrollTop = scroll[0]; nextScroll.scrollLeft = scroll[1];
}

/* ---------------------------------------------------------------- charts */
function giScale(d0, d1, r0, r1) { return (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0); }
function giQuantile(vals, q) {
  const s = [...vals].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))] : 0;
}
function giTicks(lo, hi, n = 5) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0 || 1));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}

function giScatter(opts) {
  // opts: {title, sub, pts:[{s,x,y}], xLabel, yLabel, xTicks?, xFmt, yFmt, curve?, refY?, hi:Set}
  const W = 520, H = 330, M = { l: 52, r: 16, t: 14, b: 40 };
  const xs = opts.pts.map((p) => p.x), ys = opts.pts.map((p) => p.y);
  const x0 = opts.xDomain ? opts.xDomain[0] : giQuantile(xs, 0.01), x1 = opts.xDomain ? opts.xDomain[1] : giQuantile(xs, 0.99);
  const y0 = giQuantile(ys, 0.01), y1 = giQuantile(ys, 0.99);
  const padX = (x1 - x0) * 0.06 || 1, padY = (y1 - y0) * 0.08 || 1;
  const X = giScale(x0 - padX, x1 + padX, M.l, W - M.r), Y = giScale(y0 - padY, y1 + padY, H - M.b, M.t);
  const clampX = (v) => Math.min(W - M.r, Math.max(M.l, X(v))), clampY = (v) => Math.min(H - M.b, Math.max(M.t, Y(v)));
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "gi-plot", role: "img", "aria-label": `${opts.title}. 값은 종목 비교 표에서 확인할 수 있습니다.` });
  const grid = svg("g", { class: "gi-grid-lines" }, root);
  (opts.xTicks || giTicks(x0 - padX, x1 + padX)).forEach((t) => {
    const x = X(t);
    if (x < M.l - 0.5 || x > W - M.r + 0.5) return;
    svg("line", { x1: x, x2: x, y1: M.t, y2: H - M.b }, grid);
    svg("text", { x, y: H - M.b + 16, "text-anchor": "middle", class: "gi-tick" }, root).textContent = opts.xFmt(t);
  });
  giTicks(y0 - padY, y1 + padY).forEach((t) => {
    const y = Y(t);
    svg("line", { x1: M.l, x2: W - M.r, y1: y, y2: y }, grid);
    svg("text", { x: M.l - 6, y: y + 4, "text-anchor": "end", class: "gi-tick" }, root).textContent = opts.yFmt(t);
  });
  svg("text", { x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle", class: "gi-axis" }, root).textContent = opts.xLabel;
  const yl = svg("text", { x: 14, y: (M.t + H - M.b) / 2, "text-anchor": "middle", class: "gi-axis", transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})` }, root);
  yl.textContent = opts.yLabel;
  if (opts.refY != null && Y(opts.refY) > M.t && Y(opts.refY) < H - M.b) {
    svg("line", { x1: M.l, x2: W - M.r, y1: Y(opts.refY), y2: Y(opts.refY), class: "gi-ref" }, root);
    svg("text", { x: W - M.r - 2, y: Y(opts.refY) - 5, "text-anchor": "end", class: "gi-ref-label" }, root).textContent = opts.refLabel;
  }
  if (opts.curve) {
    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const xv = x0 - padX + ((x1 - x0 + 2 * padX) * i) / 60, yv = opts.curve(xv);
      if (yv != null && Y(yv) >= M.t && Y(yv) <= H - M.b) pts.push(`${X(xv).toFixed(1)},${Y(yv).toFixed(1)}`);
    }
    if (pts.length > 1) {
      svg("polyline", { points: pts.join(" "), class: "gi-curve" }, root);
      const [lx, ly] = pts[pts.length - 1].split(",").map(Number);
      svg("text", { x: lx - 2, y: ly - 6, "text-anchor": "end", class: "gi-ref-label" }, root).textContent = opts.curveLabel;
    }
  }
  const dots = opts.pts.map((p) => ({ ...p, px: clampX(p.x), py: clampY(p.y), on: opts.hi.has(p.s.symbol) }));
  const ctx = svg("g", { class: "gi-dots-ctx" }, root), hot = svg("g", { class: "gi-dots-hot" }, root);
  dots.filter((p) => !p.on).forEach((p) => svg("circle", { cx: p.px, cy: p.py, r: 2.8 }, ctx));
  const hotDots = dots.filter((p) => p.on);
  hotDots.forEach((p) => svg("circle", { cx: p.px, cy: p.py, r: 4.5 }, hot));
  // Direct labels only for the three highest on the y-axis inside the chosen sector.
  [...hotDots].sort((a, b) => b.y - a.y).slice(0, 3).forEach((p) => {
    const t = svg("text", { x: Math.min(p.px + 7, W - M.r - 2), y: p.py - 6, class: "gi-dot-label",
      "text-anchor": p.px > W - 90 ? "end" : "start" }, root);
    t.textContent = p.s.name;
  });
  const ring = svg("circle", { r: 7, class: "gi-hover-ring", visibility: "hidden" }, root);
  const tip = $("#gi-tip");
  root.addEventListener("pointermove", (e) => {
    const m = root.getScreenCTM();
    if (!m) return;
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    let best = null, bestD = 24 * 24 * (W / root.getBoundingClientRect().width) ** 2;
    for (const p of dots) {
      const dd = (p.px - pt.x) ** 2 + (p.py - pt.y) ** 2;
      if (dd < bestD) { bestD = dd; best = p; }
    }
    if (!best) { ring.setAttribute("visibility", "hidden"); tip.hidden = true; return; }
    ring.setAttribute("cx", best.px); ring.setAttribute("cy", best.py); ring.setAttribute("visibility", "visible");
    mount(tip,
      h("div", { class: "gi-tip-name" }, best.s.name, h("span", {}, ` ${best.s.symbol} · ${best.s.sub}`)),
      h("div", {}, h("b", { class: giDir(best.y) }, opts.yFmt(best.y, 1)), ` ${opts.yLabel}`),
      h("div", {}, h("b", {}, opts.xFmt(best.x, 1)), ` ${opts.xLabel}`));
    tip.hidden = false;
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - 300)}px`;
    tip.style.top = `${e.clientY + 14}px`;
  });
  root.addEventListener("pointerleave", () => { ring.setAttribute("visibility", "hidden"); tip.hidden = true; });
  return h("figure", { class: "gi-figure" },
    h("figcaption", {}, h("b", {}, opts.title), h("span", {}, opts.sub)),
    root,
    h("div", { class: "gi-legend" },
      h("span", {}, h("i", { class: "hot" }), GI.data && giStockScopeLabel()),
      h("span", {}, h("i", { class: "ctx" }), "나머지 종목"),
      opts.note ? h("span", { class: "gi-legend-note" }, opts.note) : null));
}

function giRenderCharts() {
  const d = GI.data, a = d.anchor;
  const hi = new Set(giStockRows().map((s) => s.symbol));
  const all = Object.values(d.stocks);
  const fmt = (v, dg = 0) => `${v > 0 ? "+" : ""}${v.toFixed(dg)}%`;
  const decomp = all.filter((s) => s.crash != null && s.since_low != null).map((s) => ({ s, x: s.crash, y: s.since_low }));
  const growth = all.filter((s) => s.rev_growth != null && s[GI.lens] != null && s.rev_growth > -95)
    .map((s) => ({ s, x: Math.log10(1 + s.rev_growth / 100), y: s[GI.lens] }));
  const gTicks = [-50, 0, 100, 200, 400].map((g) => Math.log10(1 + g / 100));
  const gFmt = (v, dg = 0) => fmt((10 ** v - 1) * 100, dg);
  const missing = all.length - growth.length;
  mount($("#gi-charts"),
    h("header", {}, h("h2", {}, "하락·회복과 실적 비교"),
      h("span", { class: "sub" }, `강조: ${giStockScopeLabel()} · ${hi.size}종목 · 점 하나 = 한 종목`)),
    h("div", { class: "gi-figs" },
      giScatter({
        title: "급락기 방어력과 반등", sub: ` — 곡선 위의 종목은 ${md(a.peak_date)} 종가를 회복했습니다`,
        pts: decomp, hi, xLabel: `급락기 수익률(${md(a.peak_date)}→${md(a.low_date)})`, yLabel: `저점(${md(a.low_date)}) 이후 수익률`,
        xFmt: fmt, yFmt: fmt, curve: (x) => (x > -95 ? (100 / (1 + x / 100)) - 100 : null), curveLabel: "고점 회복선",
      }),
      giScatter({
        title: "매출 성장과 주가 수익률", sub: " — 오른쪽일수록 매출 증가율이, 위일수록 선택 기간 수익률이 높습니다",
        pts: growth, hi, xLabel: "분기 매출 증가율(전년 대비, 로그 눈금)", yLabel: `${giLensLabel(GI.lens)} 수익률`,
        xTicks: gTicks, xFmt: gFmt, yFmt: fmt, refY: d.universe[GI.lens], refLabel: "AI 전체 중앙값",
        xDomain: [Math.log10(0.5), Math.log10(8)],
        note: missing ? `실적·수익률 자료가 없거나 표시 범위를 벗어난 ${missing}종목 제외` : null,
      })));
}

/* --------------------------------------------------------------- members */
function giFundLine(s, win) {
  const parts = Object.entries(s.funds || {}).filter(([, f]) => f[win]).map(([fund, f]) => {
    const k = f[win].kind, p = f[win].pct;
    const cls = GI_BUY.has(k) ? "up" : GI_SELL.has(k) ? "down" : "";
    const mark = GI_BUY.has(k) ? "▲" : GI_SELL.has(k) ? "▼" : "·";
    return h("span", { class: cls, title: "설정·환매·주식 분할을 보정한 보유 수량 변화입니다" }, `${fund} ${mark}${k}${p != null && k !== "유지" ? ` ${p > 0 ? "+" : ""}${p.toFixed(0)}%` : ""}`);
  });
  return parts.length ? parts.reduce((acc, n, i) => (i ? [...acc, " · ", n] : [n]), []) : null;
}

function giTags(s) {
  const t = GI.data.rules.tags, out = [];
  const tag = (text, dir, tip) => out.push(h("span", { class: `gi-tag ${dir || ""}`, title: tip }, text));
  if (s.rev_growth != null && s.rev_growth >= t.rev_growth) tag("매출 급증", "up", `최근 분기 매출 증가율(전년 대비) ${giPct(s.rev_growth)}`);
  if (s.fwd_pe != null && s.fwd_pe <= t.low_pe) tag(`PER ${t.low_pe}배 이하`, "", `예상 이익 기준 주가수익비율(선행 PER) ${s.fwd_pe}배`);
  if (s.ud != null && s.ud >= t.ud) tag("상승일 거래량↑", "up", `상승일 거래량 합계 ÷ 하락일 거래량 합계 = ${s.ud}배`);
  const kinds = (win) => Object.values(s.funds || {}).map((f) => f[win]?.kind).filter(Boolean);
  if (kinds("d20").some((k) => GI_BUY.has(k))) tag("펀드 매수(20일)", "up");
  else if (kinds("since_low").some((k) => GI_BUY.has(k))) tag("펀드 매수", "up", "저점 이후");
  if (kinds("d20").some((k) => GI_SELL.has(k))) tag("펀드 매도(20일)", "down");
  const f = s.kr_flow;
  if (f && f.foreign_pct >= t.flow_pct) tag("외국인 순매수", "up", `시가총액 대비 순매수액 ${giPct(f.foreign_pct, 2)}`);
  if (f && f.foreign_pct <= -t.flow_pct) tag("외국인 순매도", "down", `시가총액 대비 순매수액 ${giPct(f.foreign_pct, 2)}`);
  if (f && f.inst_pct >= t.flow_pct) tag("기관 순매수", "up", `시가총액 대비 순매수액 ${giPct(f.inst_pct, 2)}`);
  if (f && f.inst_pct <= -t.flow_pct) tag("기관 순매도", "down", `시가총액 대비 순매수액 ${giPct(f.inst_pct, 2)}`);
  if (f && f.indiv_pct >= t.flow_pct) tag("개인 순매수", "", `시가총액 대비 순매수액 ${giPct(f.indiv_pct, 2)}`);
  if (f && f.indiv_pct <= -t.flow_pct) tag("개인 순매도", "", `시가총액 대비 순매수액 ${giPct(f.indiv_pct, 2)}`);
  const sh = s.short;
  const span = sh ? ` (${md(sh.from)}→${md(sh.to)})` : "";
  if (sh && sh.cover_pct >= t.flow_pct) tag("공매도 환매", "up", `줄어든 공매도 = 발행 주식의 ${sh.cover_pct.toFixed(2)}%${span}`);
  if (sh && sh.cover_pct <= -t.flow_pct) tag("공매도 증가", "down", `늘어난 공매도 = 발행 주식의 ${(-sh.cover_pct).toFixed(2)}%${span}`);
  if (sh && sh.float_pct != null && sh.float_pct >= t.short_float) tag(`공매도 잔고 ${t.short_float}%↑`, "", `유통 주식 대비 공매도 잔고 ${sh.float_pct.toFixed(1)}%`);
  return out;
}

function giShortCell(s) {
  const sh = s.short, c = sh.cover_pct;
  return [
    h("span", { class: giDir(c), title: "한 달 사이 줄어든 공매도 주식 수(되사들인 환매)를 발행 주식 대비 %로 나타냅니다. 늘었으면 새 공매도(매도)입니다." },
      c > 0 ? `▲ 환매 ${c.toFixed(2)}%` : c < 0 ? `▼ 추가 공매도 ${(-c).toFixed(2)}%` : "변화 없음"),
    h("div", { class: "gi-fund-20" }, `잔고 유통의 ${sh.float_pct == null ? "—" : sh.float_pct.toFixed(1) + "%"} · ${md(sh.from)}→${md(sh.to)}`),
  ];
}

function giRenderMembers() {
  const d = GI.data;
  const sort = GI.sort || { key: GI.lens, dir: -1 };
  const val = (s, k) => (k === "name" ? s.name : k === "checks" ? s.checks.filter(Boolean).length : s[k]);
  const list = giStockRows().sort((a, b) => {
    const va = val(a, sort.key), vb = val(b, sort.key);
    if (va == null) return vb == null ? 0 : 1;
    if (vb == null) return -1;
    if (typeof va === "string") return sort.dir * va.localeCompare(vb, "ko");
    return sort.dir * (va - vb);
  });
  const allCols = [
    ["checks", "조건 충족", ""], ["name", "종목", ""], ["recovery", "종목 회복 등급", "r"],
    ["since_low", "저점 이후", "r"], ["crash", "급락기", "r"],
    ["cycle", "급락 전 대비", "r"], ["r1m", "1개월", "r"], ["gap_pct", "전고점 대비", "r"],
    ["mcap_usd", "시가총액", "r"], ["impact", "1억 달러당 변동", "r"], ["ud", "상승/하락 거래량 비율", "r"],
    [null, "펀드 실매매", ""], [null, "국내 수급·미국 공매도", ""], ["rev_growth", "매출 증가율", "r"],
    ["op_margin", "영업이익률", "r"], ["fwd_pe", "선행 PER", "r"], [null, "근거", ""],
  ];
  const views = {
    strength: [1, 2, 3, 4, 5, 6, 7, 10, 16],
    flow: [1, 2, 3, 10, 11, 12, 16],
    fundamentals: [1, 2, 3, 8, 9, 13, 14, 15, 16],
    all: [1, 0, ...allCols.map((_, i) => i).slice(2)],
  };
  const indices = views[GI.metricView], cols = indices.map((i) => allCols[i]);
  const head = cols.map(([k, t, c]) => h("th", { class: c + (k === GI.lens ? " lens" : ""),
    "aria-sort": k && sort.key === k ? (sort.dir < 0 ? "descending" : "ascending") : null },
  k ? h("button", { class: "gi-sort", onclick: () => {
    GI.sort = { key: k, dir: sort.key === k ? -sort.dir : (k === "name" ? 1 : -1) };
    giRenderMembers();
  } }, t, sort.key === k ? (sort.dir < 0 ? " ↓" : " ↑") : "") : t));
  const pctCell = (v) => h("td", { class: `r num ${giDir(v)}` }, giPct(v, 1));
  mount($("#gi-members"),
    h("header", {},
      h("h2", {}, "종목 비교"),
      h("span", { class: "sub gi-scope" }, `${giStockScopeLabel()} · ${list.length}/${giStockUniverse().length}종목`)),
    h("div", { class: "gi-filters" },
      giSegment("종목 비교 범위", [["selected", "선택한 분야"], ["fields", "표시된 모든 분야", "분야 검색·등급 필터에 남은 모든 분야의 종목을 중복 없이 비교합니다."],
        ["all", "전체 AI 종목", "분야 필터와 관계없이 AI 분석 대상 전체에서 종목 등급을 비교합니다."]], GI.stockScope, (scope) => {
        GI.stockScope = scope; GI.sort = null; giRenderMembers(); giRenderCharts();
      }),
      giSegment("종목 회복 등급", giGradeOptions(giStockUniverse()), GI.stockGrade, (grade) => {
        GI.stockGrade = grade; GI.sort = null; giRenderMembers(); giRenderCharts();
      }),
      giSearch("gi-stock-search", "종목 찾기", "이 범위에서 회사명·종목 코드 검색", GI.stockQuery, (query) => {
        GI.stockQuery = query; giRenderMembers(); giRenderCharts();
      })),
    h("div", { class: "gi-table-tools" },
      giSegment("표시 지표", [["strength", "가격·회복"], ["flow", "펀드·수급"],
        ["fundamentals", "실적·규모"], ["all", "모든 지표"]], GI.metricView, (view) => {
        GI.metricView = view; GI.sort = null; giRenderMembers();
      }),
      h("span", { class: "sub" }, "열 제목을 누르면 정렬")),
    h("div", { class: "gi-scroll" }, h("table", { class: "gi-table gi-member-table" },
      h("thead", {}, h("tr", {}, head)),
      h("tbody", {}, list.map((s) => {
        const cells = [
        h("td", {}, giChecks(s)),
        h("td", { class: "gi-name" }, h("b", {}, GI.metricView !== "all" ? giChecks(s) : null, s.name),
          h("span", {}, `${s.symbol.split(".")[0]} · ${s.market} · ${s.sub}`),
          GI.level === "tour" && s.tour_fields?.length ? h("span", { class: "gi-roles", title: s.tour_fields.map((f) => `${f.parent} · ${f.label}`).join(" / ") },
            s.tour_fields.slice(0, 3).map((f) => f.label).join(" · ") + (s.tour_fields.length > 3 ? ` 외 ${s.tour_fields.length - 3}` : "")) : null),
        giStockGradeCell(s),
        pctCell(s.since_low), pctCell(s.crash), pctCell(s.cycle), pctCell(s.r1m),
        h("td", { class: `r num ${giDir(s.gap_pct)}`, title: `전고점 ${s.peak_date}` }, giPct(s.gap_pct, 1),
          h("span", { class: "gi-bucket" }, s.bucket.slice(0, 1))),
        h("td", { class: "r num", title: s.mcap_usd ? `1억 달러 = 시가총액의 ${giSmallPct(1e10 / s.mcap_usd)}` : null }, giUsd(s.mcap_usd)),
        h("td", { class: "r num", title: d.rules.impact_days && `최근 ${d.rules.impact_days}거래일 동안 하루 주가 변동폭 ÷ 그날 달러 거래대금을 1억 달러 기준으로 바꾼 값의 중앙값입니다. 클수록 같은 돈에 크게 움직입니다(오를 때와 내릴 때 모두).` }, giSmallPct(s.impact)),
        h("td", { class: `r num ${s.ud != null && s.ud >= 1 ? "up" : "down"}`, title: `최근 ${d.rules.ud_days}거래일 상승일 거래량 합계 ÷ 하락일 거래량 합계. 1배 이상이면 상승일 거래량이 더 많거나 같습니다.` }, giNum(s.ud, 2)),
        h("td", { class: "gi-fund" }, giFundLine(s, "since_low") || h("span", { class: "muted" }, "—"),
          giFundLine(s, "d20") ? h("div", { class: "gi-fund-20" }, "20일: ", giFundLine(s, "d20")) : null),
        h("td", { class: "gi-flow" }, s.kr_flow ? [
          h("span", { class: giDir(s.kr_flow.foreign_pct) }, `외국인 ${giPct(s.kr_flow.foreign_pct, 2)}`), " · ",
          h("span", { class: giDir(s.kr_flow.inst_pct) }, `기관 ${giPct(s.kr_flow.inst_pct, 2)}`),
          s.kr_flow.indiv_pct != null ? h("div", { class: "gi-fund-20" }, `개인 ${giPct(s.kr_flow.indiv_pct, 2)}`,
            Math.abs(s.kr_flow.other_pct || 0) >= GI.data.rules.tags.flow_pct
              ? h("span", { title: "개인·외국인·기관의 합이 0이 아니면 나머지 주체(기타법인·자사주·ADR 원주 전환 등)가 반대편에 선 것입니다." },
                ` · 기타 주체 ${giPct(s.kr_flow.other_pct, 2)}`) : null) : null,
          s.kr_flow.foreign_hold_pp != null ? h("div", { class: "gi-fund-20" }, `외국인 보유 비율 변화 ${s.kr_flow.foreign_hold_pp > 0 ? "+" : ""}${s.kr_flow.foreign_hold_pp.toFixed(2)}%p`) : null,
        ] : s.short ? giShortCell(s) : h("span", { class: "muted" }, s.market === "한국" ? "자료 없음" : "—")),
        h("td", { class: `r num ${giDir(s.rev_growth)}` }, giPct(s.rev_growth)),
        h("td", { class: "r num" }, s.op_margin == null ? "—" : `${s.op_margin.toFixed(0)}%`),
        h("td", { class: "r num" }, s.fwd_pe == null ? "—" : `${s.fwd_pe.toFixed(1)}배`),
        h("td", { class: "gi-tags" }, giTags(s)),
        ];
        return h("tr", { class: s.star ? "star" : "" }, indices.map((i) => cells[i]));
      }),
        !list.length ? h("tr", {}, h("td", { colspan: String(cols.length), class: "gi-empty" }, "검색어와 등급에 맞는 종목이 없습니다. 검색어를 지우거나 ‘전체 AI 종목’과 다른 등급을 선택하세요.")) : null))),
    h("p", { class: "gi-foot" }, `종목 등급은 각 회사의 회복률로 따로 계산하므로 분야 등급과 다를 수 있습니다. 분야가 겹쳐도 이 목록에서는 회사당 한 번만 표시합니다. S등급은 급락에서 빠진 폭의 ${d.rules.grades.find(([g]) => g === "S")[1]}% 이상을 되찾았다는 뜻이며, 향후 수익률 순위는 아닙니다.`));
}

/* ----------------------------------------------------------------- whale */
function giRenderWhale() {
  const d = GI.data, fw = d.fund_windows || {};
  const net = (r, a, b) => (r[a] || 0) - (r[b] || 0);
  const rows = [...giSectorRows()].sort((a, b) => net(b, "fund_buy_20", "fund_sell_20") - net(a, "fund_buy_20", "fund_sell_20")
    || net(b, "fund_buy", "fund_sell") - net(a, "fund_buy", "fund_sell"));
  const counts = (b, s) => h("span", { class: "num" }, h("span", { class: b ? "up" : "muted" }, `▲${b}`), " ",
    h("span", { class: s ? "down" : "muted" }, `▼${s}`));
  const win = (k) => (fw[k] ? `${md(fw[k][0])}→${md(fw[k][1])}` : "—");
  const sw = d.short_window;
  mount($("#gi-whale"),
    h("header", {},
      h("h2", {}, "펀드 매매·투자자 수급"),
      h("span", { class: "sub" }, `펀드 실매매 건수 · TIME 기준 기간: 저점 이후 ${win("TIME_since_low")} / 최근 20일 ${win("TIME_d20")}${sw ? ` · 미국 공매도 ${md(sw[0])}→${md(sw[1])}` : ""}`)),
    h("div", { class: "gi-scroll" }, h("table", { class: "gi-table" },
      h("thead", {}, h("tr", {},
        h("th", {}, "분야"), h("th", { class: "r" }, "펀드 저점 이후"), h("th", { class: "r" }, "펀드 최근 20일"),
        h("th", { class: "r" }, "국내 외국인"), h("th", { class: "r" }, "국내 기관"), h("th", { class: "r" }, "국내 개인"),
        h("th", { class: "r" }, "국내 종목"), h("th", { class: "r" }, "미국 공매도 환매"), h("th", { class: "r" }, "미국 종목"))),
      h("tbody", {}, rows.map((r) => h("tr", {
        class: "gi-row" + (r.key === GI.sel ? " sel" : ""), tabindex: "0", onclick: () => giSelect(r.key, true),
        onkeydown: (e) => { if (e.key === "Enter") giSelect(r.key, true); },
      },
        h("td", {}, h("b", {}, r.label), r.parent ? h("span", { class: "gi-parent" }, r.parent) : null),
        h("td", { class: "r" }, counts(r.fund_buy, r.fund_sell)),
        h("td", { class: "r" }, counts(r.fund_buy_20, r.fund_sell_20)),
        h("td", { class: `r num ${giDir(r.kr_foreign_pct)}` }, r.kr_foreign_pct == null ? "—" : giPct(r.kr_foreign_pct, 2)),
        h("td", { class: `r num ${giDir(r.kr_inst_pct)}` }, r.kr_inst_pct == null ? "—" : giPct(r.kr_inst_pct, 2)),
        h("td", { class: `r num ${giDir(r.kr_indiv_pct)}` }, r.kr_indiv_pct == null ? "—" : giPct(r.kr_indiv_pct, 2)),
        h("td", { class: "r num" }, r.kr_n ? String(r.kr_n) : "—"),
        h("td", { class: `r num ${giDir(r.us_short_cover)}` }, r.us_short_cover == null ? "—" : giPct(r.us_short_cover, 2)),
        h("td", { class: "r num" }, r.us_n ? String(r.us_n) : "—")))))),
    h("p", { class: "gi-foot" }, "▲는 매수, ▼는 매도 건수입니다. 펀드와 종목의 조합별로 세므로 두 펀드가 같은 종목을 사면 2건입니다. 설정·환매·주식 분할을 보정한 수량 변화가 ±5%를 넘는 경우를 집계합니다. 펀드 PDF는 전일 보유 내역입니다. 국내 수치는 저점 다음 날 이후 외국인·기관·개인 순매수액의 시가총액 대비 비율이며, 섹터별로 시가총액 가중 평균을 냅니다. 세 주체의 합이 0이 아니면 나머지(기타법인·자사주·ADR 원주 전환 등)가 반대편에 선 것입니다. 2026년 검증에서 지난달 국내 수급으로 다음 달 수익을 맞히지는 못했습니다. 미국은 주체별 매매가 공개되지 않아 공매도 잔고의 한 달 변화를 씁니다. 플러스는 공매도를 되사들인 환매(매수), 마이너스는 새 공매도(매도)이며 발행 주식 대비 %를 달러 시가총액으로 가중합니다."));
}

function giRenderNotes() {
  const d = GI.data, a = d.anchor;
  mount($("#gi-notes"),
    h("p", {}, `비교 기준은 ${a.index}의 최근 ${d.rules.anchor_lookback}거래일 중 하락폭이 가장 컸던 구간(${md(a.peak_date)} 고점 → ${md(a.low_date)} 저점)입니다. 모든 종목을 이 날짜에 맞춰 비교하며, 아시아 종목은 해당 날짜 또는 직전 거래일 종가를 사용합니다. ‘급락 전 대비’는 기준 고점일의 종가와 비교하고, ‘전고점 대비’는 각 종목의 역대 최고가와 비교합니다.`),
    h("p", {}, "섹터 수익률은 소속 종목의 중앙값입니다. 중앙값은 수치를 순서대로 나열했을 때 가운데에 있는 값으로, 일부 종목의 급등락 영향을 줄입니다. 매출 증가율은 최근 분기를 전년 같은 분기와 비교한 값이며, 선행 PER은 예상 이익 기준 주가수익비율입니다."),
    d.tour_sectors?.length ? h("p", {}, `‘사업 분야’는 AI 산업 지도의 사업 역할을 AI 분석 대상 안에서 연결한 ${d.tour_sectors.length}개 비교 단위입니다. 같은 회사가 여러 분야에 들어갈 수 있어 분야별 종목 수·매매 건수를 더해 전체 합계로 읽지 않습니다. ‘전체 AI 종목’ 비교에서는 중복을 없앱니다. ◆ 조건과 위 재현 검증은 ‘대표 사업 분류’를 기준으로 하며 사업 분야 등급으로 다시 검증한 결과는 아닙니다.`) : null,
    h("p", {}, "가격·실적은 Yahoo Finance, 국내 외국인·기관 매매는 네이버 증권, 펀드 실매매는 검증한 보유 내역 PDF에서 계산합니다. 자료가 없으면 ‘—’로 표시합니다. 국내 수급은 최근 60거래일 안에서 수집하므로 저점이 더 오래전이면 일부 기간만 반영됩니다."),
    h("p", {}, "◆는 주가·거래량 조건을 충족한 주도주 후보이고, AI 산업 지도의 ★는 사업 분야별 대표 기업입니다. 이 화면은 매수 추천이나 주문 수량을 제공하지 않습니다."),
    d.breadth ? h("p", {}, `검증(${d.breadth.check.period}, 매주 그 시점 자료로 ◆를 다시 고른 ${d.breadth.check.dates}개 시점): ◆ 종목의 20거래일 수익은 시장 폭이 ${d.breadth.check.split}% 이상일 때 AI 전체보다 평균 +${d.breadth.check.wide_excess}%p(적중 ${d.breadth.check.wide_hit}%), 미만일 때 ${d.breadth.check.narrow_excess}%p(적중 ${d.breadth.check.narrow_hit}%)였습니다. 지수가 바닥을 만드는 날에는 ‘저점 이후’가 모두 0이 되어 ◆가 비고, 급락 직후 반등은 많이 빠진 종목이 이끌었습니다. 시점이 적고 기간이 겹치므로 참고용입니다.`) : null,
    h("p", {}, "국내 수급의 ‘기타 주체’는 개인·외국인·기관 합의 반대편(기타법인·자사주·ADR 원주 전환 등)입니다. 이 값이 크면 외국인 순매도를 큰손의 이탈로만 해석하지 마세요."),
    d.rules.impact_days ? h("p", {}, `‘시가총액’은 Yahoo Finance 시가총액을 최근 환율로 달러 환산한 값입니다. ‘1억 달러당 변동’은 최근 ${d.rules.impact_days}거래일 동안 하루 주가 변동폭을 그날 달러 거래대금으로 나눠 1억 달러 기준으로 바꾼 값의 중앙값입니다. 클수록 같은 돈이 들어오거나 나갈 때 주가가 크게 움직입니다. 7월 급락 때 작은 종목은 더 크게 빠졌고 반등도 더 컸습니다. 실제로 1억 달러를 사면 작은 종목은 이보다 훨씬 크게 움직이거나 그만큼 살 수 없으므로 종목끼리 비교하는 데만 쓰세요.`) : null,
    h("p", {}, "미국 거래소는 개인·기관·외국인별 매매를 공개하지 않습니다. 그래서 공매도 잔고(FINRA, 한 달에 두 번 집계하고 약 2주 뒤 공개)의 한 달 변화를 보여 줍니다. 잔고가 줄면 공매도 세력이 주식을 되사들인 것이고, 늘면 새로 판 것입니다. 기관 보유(13F)는 분기 말 보유를 45일 뒤에 공개하므로 저점 이후 매매를 아직 보여 주지 못해 넣지 않았습니다."));
  let warnings = $("#gi-alerts");
  if (!warnings) {
    warnings = h("section", { class: "alerts", id: "gi-alerts", "aria-live": "polite" });
    $("#gi-kpis").before(warnings);
  }
  mount(warnings, d.warnings.length ? h("div", { class: "banner" }, "자료 확인 필요: " + d.warnings.join(" · ")) : null);
}

/* ------------------------------------------------------------------ boot */
(function giBoot() {
  if (!$("#gi-tip")) document.body.append(h("div", { class: "gi-tip", id: "gi-tip", role: "presentation", hidden: true }));
  window.addEventListener("desk:tab", (e) => {
    if (e.detail === "giant" && !GI.data && !GI.loading) giLoad(false);
  });
})();
